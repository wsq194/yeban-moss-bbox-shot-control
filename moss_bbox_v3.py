"""Compile local detector boxes and one vision response into MOSS prompts."""

import hashlib
import io
import json
import math

from PIL import Image, ImageDraw

from . import nodes
from .moss_bbox_v2_backends import detect_boxes, llama_facts, resolve_backend


_CACHE_VERSION = "moss-vision-facts-v3.3"
_REFERENCE_ASPECT = "reference_aspect"
_SHOT_LABELS = {
    "reference": "沿用原图",
    "close": "近景",
    "environmental": "中景",
    "wide": "远景",
}
_SHOT_HEIGHT_TARGETS = {"close": 0.90, "environmental": 0.58, "wide": 0.34}
_POSITION_TARGETS = {
    "upper_left": (0.15, 0.15), "upper_center": (0.50, 0.15), "upper_right": (0.85, 0.15),
    "middle_left": (0.15, 0.50), "center": (0.50, 0.50), "middle_right": (0.85, 0.50),
    "lower_left": (0.15, 0.85), "lower_center": (0.50, 0.85), "lower_right": (0.85, 0.85),
}
_POSITION_LABELS = {
    "upper_left": "upper left", "upper_center": "upper center", "upper_right": "upper right",
    "middle_left": "middle left", "center": "center", "middle_right": "middle right",
    "lower_left": "lower left", "lower_center": "lower center", "lower_right": "lower right",
}


def _apply_shot_size(layout, shot_size, composition_position, width, height, require_full_figure=True):
    summary = layout["summary"]
    items = layout["items"]
    if shot_size == "reference":
        summary["caption"] = (
            f"Preserve the reference composition and subject scale. {summary['caption']}"
        )
        nodes._moss_describe_character_framing(layout, width, height)
        return

    if summary.get("canvas", "single") != "single":
        raise nodes.PromptWriterError("近景、中景和远景适用于单画面，请为多面板布局选择沿用原图")

    background, *foreground = items
    characters = [item for item in foreground if item["label"] == "character"]
    if not characters:
        raise nodes.PromptWriterError("近景、中景和远景需要人物框")

    max_character_height = max(item["bbox"][2] - item["bbox"][0] for item in characters)
    if max_character_height <= 0:
        raise nodes.PromptWriterError("人物框高度无效，无法计算景别布局")

    target_height = _SHOT_HEIGHT_TARGETS[shot_size]
    scale = height * target_height / max_character_height
    character_left = min(item["bbox"][1] for item in characters)
    character_right = max(item["bbox"][3] for item in characters)
    max_character_width = character_right - character_left
    if max_character_width > 0:
        scale = min(scale, width * 0.92 / max_character_width)

    for item in foreground:
        top, left, bottom, right = item["bbox"]
        item["bbox"] = [round(top * scale), round(left * scale), round(bottom * scale), round(right * scale)]

    # Keep the character group inside the sampling canvas. The original composition
    # may put a source-cropped person close to an edge; enlarging around the canvas
    # origin alone can push that box outside the model's editable image area.
    frame_top, frame_left = -height // 2, -width // 2
    frame_bottom, frame_right = frame_top + height, frame_left + width
    character_top = min(item["bbox"][0] for item in characters)
    character_left = min(item["bbox"][1] for item in characters)
    character_bottom = max(item["bbox"][2] for item in characters)
    character_right = max(item["bbox"][3] for item in characters)
    position_x, position_y = _POSITION_TARGETS.get(composition_position, _POSITION_TARGETS["center"])
    target_x = frame_left + width * position_x
    target_y = frame_top + height * position_y
    requested_shift_x = target_x - (character_left + character_right) / 2
    requested_shift_y = target_y - (character_top + character_bottom) / 2
    margin_x, margin_y = width * 0.03, height * 0.03
    min_shift_x = frame_left + margin_x - character_left
    max_shift_x = frame_right - margin_x - character_right
    min_shift_y = frame_top + margin_y - character_top
    max_shift_y = frame_bottom - margin_y - character_bottom
    shift_x = min(max(requested_shift_x, min_shift_x), max_shift_x)
    shift_y = min(max(requested_shift_y, min_shift_y), max_shift_y)

    visible_items = []
    for item in foreground:
        top, left, bottom, right = item["bbox"]
        top = max(frame_top, top + shift_y)
        left = max(frame_left, left + shift_x)
        bottom = min(frame_bottom, bottom + shift_y)
        right = min(frame_right, right + shift_x)
        if top >= bottom or left >= right:
            if item["label"] == "character":
                raise nodes.PromptWriterError("景别布局将人物框移出画布，无法生成有效的 MOSS 坐标")
            continue
        item["bbox"] = [top, left, bottom, right]
        visible_items.append(item)
    layout["items"] = [
        background,
        *(item for item in visible_items if item["label"] != "character"),
        *(item for item in visible_items if item["label"] == "character"),
    ]

    placement = []
    for index, item in enumerate((item for item in layout["items"] if item["label"] == "character"), 1):
        top, left, bottom, right = item["bbox"]
        center_x = (left + right) / (2 * width) + 0.5
        center_y = (top + bottom) / (2 * height) + 0.5
        placement.append(
            f"Person {index} box: {bottom - top} pixels tall ({(bottom - top) / height:.0%} of frame height), "
            f"centered at {center_x:.0%} from the left and {center_y:.0%} from the top."
        )

    shot_text = {
        "close": (
            "Close shot: keep the person large at about 90% of the image height."
        ),
        "environmental": (
            "Medium shot: keep the person at about 58% of the image height, with the nearby setting clearly visible."
        ),
        "wide": (
            "Wide shot: keep the person small at about 34% of the image height; let the surrounding scene dominate."
        ),
    }[shot_size]
    position_text = _POSITION_LABELS.get(composition_position, "center")
    subject_lock = (
        "Change only the person's scale, selected placement, and the amount of visible surrounding space. "
        "Preserve the source pose, body orientation, expression, clothing layers, support contact and visible crop; "
        "do not change a seated pose into a standing pose or invent body parts hidden by the source."
    )
    crop_rule = (
        "Keep only body parts established by the reference; do not invent anatomy hidden by its crop. "
        if not require_full_figure else "Keep the full figure visible with ground below the feet. "
    )
    summary["caption"] = (
        f"{shot_text} {subject_lock} {crop_rule} "
        f"Place the person group and its foreground layout in the {position_text} area of the frame, "
        "while preserving the stated relationships between the person and nearby objects. "
        f"{' '.join(placement)} {background['caption']} "
        "The continuous scene reaches all four image edges with no border or blank margin."
    )


def _boxes(value, width, height, prefix):
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise nodes.PromptWriterError("检测框格式错误，请接入单张参考图对应的本地检测框。")
    boxes = []
    for box in value:
        try:
            x, y, w, h = (float(box[key]) for key in ("x", "y", "width", "height"))
            if not all(math.isfinite(number) for number in (x, y, w, h, x + w, y + h)):
                raise ValueError
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise nodes.PromptWriterError("检测框必须包含有效的 x、y、width、height。") from exc
        bounded = [max(0, x), max(0, y), min(width, x + w), min(height, y + h)]
        if bounded[0] < bounded[2] and bounded[1] < bounded[3] and bounded not in boxes:
            boxes.append(bounded)
    boxes.sort(key=lambda box: (box[0] + box[2], box[1], *box))
    return {f"{prefix}{i}": box for i, box in enumerate(boxes, 1)}


def _facts(raw, fields):
    for value in nodes._candidate_json_values(raw):
        if isinstance(value, dict) and set(value) == set(fields) and all(
            isinstance(item, str) and (item.strip() or key.startswith("O")) for key, item in value.items()
        ):
            return {key: item.strip() for key, item in value.items()}
    raise nodes.PromptWriterError("反推后端返回的编号描述或智能反推不完整，请重试。")


class YebanMossBBoxStandaloneV3(nodes.YebanMossBBoxPrompt):
    """Cache source-image facts independently from local shot-size transforms."""

    @classmethod
    def INPUT_TYPES(cls):
        parent = super().INPUT_TYPES()
        v1 = parent["required"]
        return {"required": {
            "image": v1["image"],
            "layout_aspect": ([*v1["layout_aspect"][0], _REFERENCE_ASPECT], v1["layout_aspect"][1]),
            "layout_long_edge": ("INT", {
                **v1["layout_long_edge"][1],
                "default": 3072,
            }),
            "shot_size": (list(_SHOT_LABELS), {
                "default": "reference",
                "display_name": "景别",
                "tooltip": "景别同时控制人物布局比例：近景约占画高90%，中景约58%，远景约34%；沿用原图保留原始比例。",
            }),
            "backend": v1["backend"],
            "backend_model": v1["backend_model"],
            "custom_model": v1["custom_model"],
            "reasoning": (["low"], {**v1["reasoning"][1], "default": "low"}),
            "subject_setting": v1["subject_setting"],
            "photography_style": v1["photography_style"],
            "additional_requirements": v1["additional_requirements"],
            "timeout": v1["timeout"],
            "use_cache": v1["use_cache"],
            "cache_seed": v1["cache_seed"],
            "target_word_count": ("INT", {"default": 220, "min": 50, "max": 1500, "display_name": "智能反推目标英文词数"}),
            # Keep this new widget at the end so saved V3 widget values retain
            # their existing indices when the position selector is added.
            "composition_position": (list(_POSITION_TARGETS), {
                "default": "center",
                "display_name": "人物位置",
                "tooltip": "选择人物与关联前景道具在画面中的位置；沿用原图时不改变位置，近景上下移动会受安全边距限制。",
            }),
        }, "optional": {key: parent["optional"][key] for key in (
            "provider_config", "dsh_config", "llama_model", "llama_parameters",
        )}}

    RETURN_TYPES = nodes.YebanMossBBoxPrompt.RETURN_TYPES
    RETURN_NAMES = ("MOSS 布局提示词", "采样宽度", "采样高度", "智能反推提示词")
    FUNCTION = "write_layout"
    CATEGORY = "Yeban/MOSS BBox"

    @nodes.official_dsh.scoped
    def write_layout(self, image, layout_aspect, layout_long_edge, shot_size,
                     backend, backend_model, custom_model, reasoning,
                     subject_setting, photography_style, additional_requirements, timeout, use_cache,
                     cache_seed, target_word_count, composition_position="center", provider_config=None, dsh_config=None,
                     llama_model=None, llama_parameters=None):
        frames = nodes._normalise_rgb_batch(image)
        if len(frames) != 1:
            raise nodes.PromptWriterError("MOSS BBox V3 每次处理一张参考图。")
        photo = Image.fromarray(frames[0])
        source_w, source_h = photo.size
        model, base_url, api_key, identity = resolve_backend(
            backend, str(custom_model or backend_model or "").strip(),
            provider_config, dsh_config, llama_model, llama_parameters,
        )
        person_boxes, prop_boxes = detect_boxes(frames[0], use_cache)
        people = _boxes(person_boxes, source_w, source_h, "P")
        if not people:
            raise nodes.PromptWriterError("本地检测没有找到人物框，请检查参考图中是否有清晰可见的人物。")
        boxes = {**people, **_boxes([] if prop_boxes is None else prop_boxes, source_w, source_h, "O")}
        if layout_aspect == _REFERENCE_ASPECT:
            if type(layout_long_edge) is not int or not 0 <= layout_long_edge <= 16384 or layout_long_edge % 16:
                raise nodes.PromptWriterError("采样长边须为 0 或不超过 16384 的 16 像素倍数")
            source_edge = max(source_w, source_h)
            long_edge = layout_long_edge or min(16384, max(16, round(source_edge / 16) * 16))
            width = max(16, round(source_w * long_edge / source_edge / 16) * 16)
            height = max(16, round(source_h * long_edge / source_edge / 16) * 16)
        else:
            aspect = nodes._normalise_krea2_bbox_aspect(layout_aspect)
            width, height = nodes._moss_layout_dimensions(aspect, image, layout_long_edge)
        draw = ImageDraw.Draw(photo)
        stroke = max(1, round(min(photo.size) / 400))
        for key, (x0, y0, x1, y1) in boxes.items():
            color = "yellow" if key in people else "cyan"
            draw.rectangle((x0, y0, min(x1, source_w - 1), min(y1, source_h - 1)), outline=color, width=stroke)
        for key, (x0, y0, x1, y1) in boxes.items():
            color = "yellow" if key in people else "cyan"
            anchor_y = y0 + (y1 - y0) * (0.09 if key in people else 0.5)
            draw.text(((x0 + x1) / 2, anchor_y), key, anchor="mm", fill=color,
                      stroke_width=stroke, stroke_fill="black",
                      font_size=max(12, round(min(photo.size) / 30)))
        buffer = io.BytesIO()
        photo.save(buffer, format="PNG")
        annotated_png = buffer.getvalue()
        crop_fields = {key: f"{key}_source_crop" for key in people}
        fields = [*boxes, *crop_fields.values(), "background_caption", "smart_reverse_prompt"]
        schema = {"type": "object", "required": fields, "additionalProperties": False,
                  "properties": {field: {"type": "string"} for field in fields}}
        crop_fact_instruction = (
            "For each person, also provide a separate source-crop field named "
            + ", ".join(f"{field} for {person}" for person, field in crop_fields.items())
            + ". State only how the original frame cuts the visible body or garment (for example, feet cut by the bottom edge, "
            "or the full figure visible with margin). This field is used only for 沿用原图; do not include crop wording in the person caption. "
            "Do not infer hidden anatomy. "
        )
        framing_rule = (
            "Describe visible pose, clothing, props, and scene facts from the reference without prescribing the source crop in person item captions. "
            "Do not invent hidden anatomy. The smart_reverse_prompt preserves the original image crop; person item captions remain crop-neutral "
            "because the selected output framing is applied afterward."
        )
        brief = (
            f"Inspect the marked reference photograph once. People: {', '.join(people)}. "
            f"Key props: {', '.join(key for key in boxes if key not in people) or 'none marked'}. "
            "For each person ID, write one compact English layout caption of 50 to 90 words. Start with the clearest pose and "
            "body-support relation, then the torso's lean/recline separately from which way the chest and head face, followed by "
            "visible hand/limb placement. Do not reduce a sideways lean or backward recline to merely 'facing the camera'. State the source-visible "
            "visible body and garment details, but do not describe the source frame edges, crop, or anything continuing beyond the image. "
            "Finish with the exact visible garment layers and colors. Do not lead with age, identity, or crop when a clear pose is visible. "
            "Do not repeat the same fact in multiple phrases. Keep each person and prop ID compact. For each prop ID, use one compact caption of at most 35 words. "
            "Match each ID to the main subject across its entire rectangle, using the label as a guide. "
            "Ignore people printed on posters. If a marked prop is a false detection or part of a person, return an empty string for its ID. "
            "The local detector owns the geometry. "
            "Pose accuracy is a priority. Give an explicit posture category whenever the visible body alignment supports one, then state "
            "the body-to-surface support relation. When the visible cues show the pelvis and upper thigh bearing weight on a horizontal "
            "seat, ledge, or counter, explicitly say the person is seated on that surface; you may add 'perched at the edge' as a modifier, "
            "but never use 'perched' alone or describe only a hip touch. Use standing/leaning only when the legs bear the body weight "
            "and the nearby surface is "
            "not supporting the pelvis. If the visible geometry cannot resolve support, state that uncertainty consistently in both outputs. "
            "Then describe how the torso and shoulders lean/recline and turn relative to the camera as separate facts. "
            "Compare the visible pelvis, waist/rib cage, shoulder midpoint, and head positions: when the torso axis or shoulder line is diagonal, "
            "state its direction using image-relative terms (toward the left/right side of the image, upward/downward), and describe head tilt/orientation separately. "
            "Do not flatten a clearly diagonal torso into 'nearly upright' or replace its lean with a generic camera-facing description; do not force a lean when the visible axis is upright. "
            "Before naming a diagonal, explicitly compare the head center with the pelvis center (which is farther left/right in the image) and compare the image-left shoulder with the image-right shoulder (which is higher). "
            "Use frame-relative positions, not the person's anatomical left/right. Avoid ambiguous words such as 'inward' or 'leaning toward camera' when a left/right relation is visible. "
            "Cross-check these exact head-to-pelvis and shoulder-height relations against the reference image and keep them consistent between the layout caption and smart_reverse_prompt. "
            "Keep torso direction and shoulder-line tilt as distinct facts, then each visible hand/arm position and leg placement. Avoid anatomical left/right labels for hands when "
            "the side is ambiguous; use 'the visible hand' and 'the other hand/arm' or a clear image-relative location so the hand on "
            "the skirt is not swapped to the nearby furniture. When a visible hand rests on the person's own skirt or lap near furniture, "
            "say that garment contact explicitly and distinguish it from a hand resting on the furniture. Describe each arm separately: "
            "where its elbow/forearm folds and where the hand rests. If a hand is hidden by a sleeve, state the visible sleeve/forearm "
            "route and where it disappears without inventing fingers; do not leave the arm's route unspecified when its contour shows it "
            "folding toward the lap. If the only visible hand rests on the person's skirt and no hand touches the furniture, make clear "
            "that the furniture supports the hip/body only, not a hand. Explicitly distinguish a hand touching or resting on a surface from the hips or pelvis being "
            "supported there. Do not treat clothing occlusion by itself as evidence that support is unknown. Inspect how the torso leans "
            "relative to a nearby seat, ledge, or counter, where the hip/upper thigh aligns with its top edge, and whether the skirt or "
            "clothing drapes down from that contact. When multiple visible cues indicate that the pelvis or hip rests on the surface, "
            "include the explicit seated-on-surface category in both the person caption and smart_reverse_prompt even if the exact contact "
            "point is partly hidden and the lower legs are cropped; do not omit the posture category or mislabel it as standing because "
            "the feet are unseen. Leave support uncertain only when the visible body alignment "
            "and surface geometry do not establish likely contact. Do not infer support from a hand alone or mere proximity to a counter. "
            "Prioritize both the explicit posture category and exact contact wording. Describe visible people accurately in their ID fields, including torso angle, support "
            "contact, hand placement, clothing, and visible anatomy. Before finalizing, cross-check every person ID caption against "
            "smart_reverse_prompt for torso angle, support relation or uncertainty, hand placement, visible clothing layers and exposed skin. "
            "The smart_reverse_prompt must preserve the reference crop, while the person item caption must not dictate output crop or frame-edge contact. Do not turn unresolved "
            "support into a confident standing, seated, or perched claim in either output. "
            f"{crop_fact_instruction}{framing_rule} "
            "Do not invent body parts hidden by the crop. Attach each clothing color to the correct garment; preserve "
            "distinct layers, outerwear, sheer fabric and accessories. Mention a garment hem only if it is actually visible. If the source crop hides the hem, "
            "record that uncertainty for smart_reverse_prompt, but do not say the garment continues beyond an image edge in the person item caption. "
            "Do not shorten a visible long/tiered outer layer or infer exposed knees/lower legs from cropped areas; mention skin only where clearly visible. "
            "Keep the source crop in smart_reverse_prompt and keep person item captions free of output frame-edge instructions. In background_caption describe only the continuous "
            "environment, foreground and surfaces reaching the edges; do not repeat or reinterpret the person. "
            "For objects and surfaces, use visible material cues and keep the chosen material consistent in the global caption, item caption, "
            "and smart_reverse_prompt. Describe observable color, reflection, grain, and seams instead of guessing a material when uncertain. "
            "Do not call a cool-gray reflective worktop marble without visible stone veining when its smooth directional reflections and seams "
            "indicate brushed metal. Do not invent a rug or crumpled cloth from a continuous hard floor's mottled reflections; require clear "
            "fabric edges and folds before naming a separate soft floor covering. "
            f"In smart_reverse_prompt write a standalone English image prompt of about {target_word_count} words, "
            "preserving visible clothing construction, expression, pose, original crop, lighting and environment. "
            + " "
            "Keep identifiers, coordinates, and analysis out of the prose. "
            "Apply supplied settings when relevant while preserving observed pose and clothing; the output crop is controlled separately by the selected shot size. "
            f"Subject setting: {subject_setting}\nPhotography style: {photography_style}\n"
            f"Additional requirements: {additional_requirements}"
        )
        digest = hashlib.sha256(annotated_png)
        digest.update(json.dumps([_CACHE_VERSION, boxes, brief, backend, model, reasoning, cache_seed, identity],
                                 ensure_ascii=False).encode("utf-8"))
        key = digest.hexdigest()
        cache_facts = use_cache
        cached = nodes._read_cache(key, schema_version=_CACHE_VERSION) if cache_facts else None
        if cached:
            try:
                facts = _facts(cached["prompt_text"], fields)
            except nodes.PromptWriterError:
                cached = None
        if not cached:
            instruction = brief + "\nReturn one JSON object with these string fields: " + ", ".join(fields)
            if backend == "local_llama":
                raw = llama_facts(instruction, annotated_png, schema, llama_model, llama_parameters, cache_seed)
            else:
                raw = nodes._call_backend(
                    instruction, annotated_png, backend=backend, base_url=base_url, model=model,
                    reasoning=nodes._normalise_reasoning(reasoning), timeout=timeout,
                    api_key_override=api_key, output_schema=schema,
                )
            facts = _facts(nodes._response_text_for_repair(raw), fields)
            if cache_facts:
                nodes._write_cache(key, {"prompt_text": json.dumps(facts, ensure_ascii=False)},
                                   schema_version=_CACHE_VERSION)

        draft = {
            "width": source_w, "height": source_h, "bbox_coord_space": "abs", "bbox_format": "xyxy",
            "summary": {"caption": facts["background_caption"], "canvas": "single"},
            "items": [{
                "label": "character" if key in people else "object",
                "caption": facts[key] + (
                    f" Original source framing: {facts[crop_fields[key]]}"
                    if shot_size == "reference" and key in crop_fields else ""
                ),
                "bbox": box,
            }
                      for key, box in boxes.items() if facts[key]],
            "background_caption": facts["background_caption"],
        }
        layout = json.loads(nodes._moss_layout_draft_to_prompt(
            json.dumps(draft, ensure_ascii=False), width, height,
            sampling_width=width, sampling_height=height, source_size=(source_w, source_h),
        ))
        # Avoid requiring unseen feet when a reference subject reaches the bottom edge.
        cropped_reference = any(
            box[3] >= source_h * 0.98 for box in people.values()
        )
        _apply_shot_size(
            layout, shot_size, composition_position, width, height,
            require_full_figure=not cropped_reference,
        )
        sampling_prompt = json.dumps(layout, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return sampling_prompt, width, height, facts["smart_reverse_prompt"]

