// Project-side adapter; never modifies the installed official application.
import path from 'node:path';
import os from 'node:os';
import {pathToFileURL} from 'node:url';
const install=process.env.DSH_OFFICIAL_INSTALL||path.join(os.homedir(),'AppData/Local/Programs/DeepSeek Harness');
const base=path.join(install,'resources/app.asar/dsh/node_modules/@deepseek-ai');
const load=name=>import(pathToFileURL(path.join(base,name,'lib/index.js')).href);

if(process.argv.includes('--fork')||process.argv.includes('--list')) {
  const [sourceId,childId,title]=process.argv.slice(process.argv.indexOf('--fork')+1);
  if(!process.argv.includes('--list')&&(![sourceId,childId].every(id=>/^[A-Za-z0-9_-]{1,100}$/.test(id))||sourceId===childId))throw new Error('DSH_OFFICIAL_BRANCH_INVALID');
  const {Context}=await load('cordis');
  const {buildForkSeed}=await load('dsh-session');
  const ctx=new Context();
  const home=process.env.DSH_OFFICIAL_HOME||process.env.DSH_HOME||path.join(os.homedir(),'.dsh');
  try {
    await ctx.plugin((await load('dsh-storage')).default).await();
    await ctx.plugin(await load('dsh-storage-json'),{root:path.join(home,'storages')}).await();
    await ctx.plugin(await load('dsh-storage-domain'),{backend:'json'}).await();
    await ctx.plugin((await load('dsh-session')).default).await();
    const jsonlPersistence=await load('dsh-session-persistence-jsonl');
    await ctx.plugin(jsonlPersistence.default??jsonlPersistence,{root:path.join(home,'sessions'),compression:'zstd'}).await();
    await ctx.plugin((await load('dsh-workspace')).default).await();
    if(process.argv.includes('--list')) {
      const persistence=ctx.get('sessionPersistence'),rows=[];
      const registered=new Set(ctx.get('workspaceRegistry').list().flatMap(w=>w.sessionIds));
      const snapshots=await persistence.list();
      for(const snapshot of snapshots) {
        const header=snapshot.header;
        if(!registered.has(header.id))continue;
        let title='未命名对话';
        const session=await persistence.open(header.id,'read');
        try {for(const event of (await session.read(0)).events)if(event.type==='session/title'&&event.data?.title)title=event.data.title;}finally{await session.close();}
        if(header.parentSession)title=title.replace(/ · \d{4}-\d{2}-\d{2} \d{2}:\d{2}$/,'');
        rows.push({id:header.id,title,created_at:new Date(header.createdAt).toISOString(),parent_session:header.parentSession||''});
      }
      rows.sort((a,b)=>b.created_at.localeCompare(a.created_at));
      console.log(JSON.stringify(rows));
    }else{
    const workspace=ctx.get('workspaceRegistry').list().find(w=>w.sessionIds.includes(sourceId));
    if(!workspace)throw new Error('DSH_OFFICIAL_SOURCE_WORKSPACE_MISSING');
    const persistence=ctx.get('sessionPersistence');
    const existing=(await persistence.list()).find(snapshot=>snapshot.header.id===childId);
    if(existing){
      if(existing.header.parentSession!==sourceId)throw new Error('DSH_OFFICIAL_BRANCH_PARENT_MISMATCH');
      const stored=await persistence.open(childId,'read');
      try{await stored.read(0);}finally{await stored.close();}
      await workspace.attachSession(childId);
      console.log(JSON.stringify({sessionId:childId,parentSession:sourceId,created_at:new Date(existing.header.createdAt).toISOString(),cwd:existing.header.cwd}));
    }else{
    const source=await persistence.open(sourceId,'read');
    let events,header;
    try {events=(await source.read(0)).events;header=source.header;}finally {await source.close();}
    const seed=events.length?buildForkSeed(events,events.length-1):[];
    const child=await persistence.create({...header,id:childId,createdAt:Date.now(),cwd:header.cwd,parentSession:sourceId,isSeeded:true,agentPreset:header.agentPreset},{inheritedEventCount:events.length});
    const createdAt=child.header.createdAt;
    const storedTitle=process.argv.includes('--timestamp-title')?title.replace(/ · \d{4}-\d{2}-\d{2} \d{2}:\d{2}$/,'')+' · '+new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(createdAt)):title;
    try {
      await child.append([...seed,{type:'session/title',seq:seed.length,time:Date.now(),data:{title:storedTitle,source:{kind:'user'},messageSeqs:[]}}]);
    }finally {await child.close();}
    await workspace.attachSession(childId);
    console.log(JSON.stringify({sessionId:childId,parentSession:sourceId,inheritedEventCount:events.length,created_at:new Date(createdAt).toISOString(),cwd:header.cwd}));
    }
    }
  }finally {await ctx.fiber.dispose();}
} else if(process.env.STORYBOARD_DSH_BRANCH_SESSION) {
  const branchId=process.env.STORYBOARD_DSH_BRANCH_SESSION;
  const {HarnessSdkJsonRpcServer}=await load('dsh-sdk-jsonrpc-server');
  const original=HarnessSdkJsonRpcServer.prototype.createSession;
  HarnessSdkJsonRpcServer.prototype.createSession=async function(sessionId) {
    if(sessionId!==branchId)return original.call(this,sessionId);
    const handle=await this.ctx.agents.resume({resumeSessionId:sessionId,agentOptions:{provider:this.provider,model:this.model,...this.reasoningEffort===undefined?{}:{reasoningEffort:this.reasoningEffort},...this.maxTokens===undefined?{}:{maxTokens:this.maxTokens}}});
    const rec={handle};this.sessions.set(sessionId,rec);return rec;
  };
}
