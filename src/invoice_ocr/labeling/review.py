"""Local image-invoice annotation server. One annotation and review state per invoice."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import os
import time
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
import secrets
import threading
from urllib.parse import unquote, urlsplit
from invoice_ocr.config import ROOT
from invoice_ocr.storage.files import loads_exact as loads, dumps_exact as dumps, read_jsonl, write_jsonl, images_digest
from invoice_ocr.contracts.invoice import validate_invoice
from invoice_ocr.labeling.export import reviewed_records
from invoice_ocr.labeling.comparison import refresh_verification, reset_for_verification


REVIEW_HTML = r'''<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gán nhãn hóa đơn từ ảnh</title>
<style>
*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,sans-serif;background:#f3f6f8;color:#203442}header{padding:16px 22px;background:white;border-bottom:1px solid #d7e0e5}h1{font-size:22px;margin:0 0 5px}.hint{color:#61717e;font-size:13px}button,select,input{font:inherit;padding:8px 10px;border:1px solid #bdcdd8;border-radius:6px;background:white;color:inherit}button{cursor:pointer}button:disabled{opacity:.5;cursor:wait}button.primary{background:#116b70;color:white;border-color:#116b70}.toolbar{display:flex;flex-wrap:wrap;gap:8px;padding:12px 22px;background:white}#doc{flex:1;min-width:250px;max-width:560px}#counts{padding:6px 22px 12px;color:#61717e}main{display:grid;grid-template-columns:49% 51%;border-top:1px solid #d7e0e5}.panel{padding:15px;min-width:0}.panel:first-child{border-right:1px solid #d7e0e5}.row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px}.viewer{height:70vh;overflow:auto;background:#dce3e8;border:1px solid #c2d0d9;border-radius:6px}#preview{display:block;width:100%;height:auto;cursor:zoom-in}#json{width:100%;height:63vh;resize:vertical;border:1px solid #bdcdd8;border-radius:6px;padding:12px;font:12px/1.6 Consolas,monospace;white-space:pre}#status{border-radius:20px;padding:5px 10px;background:#fff0c9}#status.reviewed{background:#d9f0e1;color:#175c32}#message{white-space:pre-wrap;color:#116b70;min-height:25px}#message.error{color:#ad2424}#name{overflow-wrap:anywhere}details{background:white;border:1px solid #ccd7de;border-radius:6px;padding:10px;margin-top:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:350px;overflow:auto;font:12px/1.5 Consolas,monospace}#empty{padding:25px}a{color:#146c89}@media(max-width:1000px){main{grid-template-columns:1fr}.viewer{height:60vh}#json{height:65vh}}
.comparison{background:white;border:1px solid #ccd7de;border-radius:6px;padding:12px;margin:10px 0}.diff-scroll{max-height:30vh;overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #d7e0e5;padding:7px;text-align:left;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}th{background:#e7f1f2}#candidateJson{max-height:40vh}#verificationNote{white-space:pre-wrap}.field-actions{display:flex;gap:5px;min-width:160px}.field-actions button{font-size:12px;padding:5px 8px}.field-actions button[aria-pressed="true"]{background:#116b70;color:white;border-color:#116b70}</style>
</head>
<body>
<header><h1>Gán nhãn hóa đơn từ ảnh</h1><div class="hint">Nhãn hiện tại → kiểm tra bằng model độc lập → khớp toàn bộ: tự reviewed; có lệch/lỗi: bạn đối chiếu ảnh và xác nhận.</div></header>
<div class="toolbar">
<select id="split" aria-label="Tập"><option>train</option><option>valid</option><option>test</option></select>
<input id="search" placeholder="Tìm mã / vendor / tên gốc">
<select id="filter"><option value="attention" selected>Có lệch / lỗi — cần kiểm tra</option><option value="pending">Nhãn hiện tại — chờ bộ kiểm tra độc lập</option><option value="unlabeled">Chưa có nhãn ban đầu</option><option value="agreed">Hai kết quả khớp — tự reviewed</option><option value="reviewed">Đã reviewed — tất cả</option><option value="needs_review">Chưa reviewed — tất cả</option><option value="all">Tất cả</option></select>
<button id="prev">←</button><select id="doc" aria-label="Hóa đơn"></select><button id="next">→</button>
<button id="reload">Tải trạng thái mới</button><button id="export">Xuất nhãn tập này</button><button id="import">Nhập JSONL</button><input id="importFile" type="file" accept=".jsonl" hidden>
</div>
<div id="counts"></div><div id="empty" hidden>Không có hóa đơn phù hợp.</div>
<main id="main" hidden>
<section class="panel">
<div class="row"><label>Ảnh <select id="image"></select></label><button id="imagePrev">← Ảnh</button><button id="imageNext">Ảnh →</button><a id="original" target="_blank" rel="noopener">Mở ảnh gốc</a></div>
<div id="name" class="hint"></div>
<div class="viewer"><img id="preview" alt="Ảnh hóa đơn"></div>
</section>
<section class="panel">
<div class="row"><span id="status"></span><span id="dirty" class="hint"></span></div>
<div class="hint">Số không có ngoặc kép; ngày YYYY-MM-DD. ID/PO/VIN/unit là chuỗi; odometer_unit: "km", "mi" hoặc null. Nhãn chung gồm toàn bộ dòng hàng và tổng của hóa đơn.</div>
<section id="verificationPanel" class="comparison">
<strong id="verificationTitle"></strong><p id="verificationNote" class="hint"></p>
<div class="diff-scroll"><table id="diffTable"><thead><tr><th>Trường lệch</th><th>Model A — nhãn hiện tại</th><th>Model B — kiểm tra độc lập</th><th>Chọn giá trị</th></tr></thead><tbody id="diffBody"></tbody></table></div>
<details id="candidateDetails"><summary>Toàn bộ kết quả bộ kiểm tra độc lập</summary><pre id="candidateJson"></pre><button id="useModel2">Chép kết quả bộ kiểm tra độc lập vào bản sửa</button></details>
</section><div class="hint">Nhãn hiện tại — sửa và bấm Cập nhật để xác nhận thủ công.</div>
<textarea id="json" spellcheck="false" aria-label="JSON nhãn hóa đơn"></textarea>
<div class="row"><button id="unreview">Bỏ reviewed</button><button id="update" class="primary">Cập nhật</button></div><div id="message" role="status"></div>

</section>
</main>
<script>
'use strict';
const $=id=>document.getElementById(id);
function parseJSON(text){return JSON.parse(text,(key,value,context)=>{if(typeof value==='number'){if(!context?.source||typeof JSON.rawJSON!=='function')throw Error('Cần Chrome/Edge mới để giữ chính xác số.');return JSON.rawJSON(context.source);}return value;});}
const clone=value=>parseJSON(JSON.stringify(value));
let records=[],visible=[],current=null,imageIndex=0,token='',busy=false;
const fieldChoices=new Map();
const key=r=>'invoice-groundtruth:invoice-v3:'+r.document_id;
function message(text,error=false){$('message').textContent=text;$('message').className=error?'error':'';}
function persist(r){try{localStorage.setItem(key(r),JSON.stringify(r));}catch(e){message('Bộ nhớ trình duyệt đầy. Bấm Cập nhật hoặc xuất JSONL để giữ bản sửa.',true);}}
function autoLabeled(r){return r.auto_labeled===true||r.review?.action==='gemini_draft'||String(r.review?.model||'').startsWith('gemini-');}
function phase(r){if(r.review_status==='reviewed')return r.review?.action==='automatic_agreement'?'Hai kết quả khớp · tự reviewed':'Đã review thủ công';return ({pending:'Nhãn hiện tại · chờ bộ kiểm tra độc lập',unlabeled:'Chưa có nhãn ban đầu',agreed:'Hai kết quả khớp · chờ xác nhận lại',disagreed:'Có lệch · cần bạn kiểm tra',error:'Bộ kiểm tra độc lập lỗi · cần xử lý'})[r.verification?.status]||'Nhãn hiện tại · chờ bộ kiểm tra độc lập';}
function matchesFilter(r,f){if(f==='all')return true;if(f==='reviewed'||f==='needs_review')return r.review_status===f;if(f==='agreed')return r.review_status==='reviewed'&&r.review?.action==='automatic_agreement';if(r.review_status==='reviewed')return false;const v=r.verification?.status||'pending';if(f==='attention')return ['disagreed','error'].includes(v)||r.review?.action==='user_unreview_invoice';return v===f;}
function counts(){const rs=records.filter(r=>r.split===$('split').value);$('counts').textContent=rs.length+' hóa đơn · '+rs.filter(r=>matchesFilter(r,'pending')).length+' chờ bộ kiểm tra độc lập · '+rs.filter(r=>matchesFilter(r,'attention')).length+' cần kiểm tra · '+rs.filter(r=>matchesFilter(r,'agreed')).length+' tự reviewed · '+rs.filter(r=>r.review_status==='reviewed'&&r.review?.action!=='automatic_agreement').length+' review thủ công';}
function badge(){if(!current)return;$('status').textContent=phase(current);$('status').className=current.review_status;const option=Array.from($('doc').options).find(o=>o.value===current.document_id);if(option)option.textContent=label(current);counts();}
function label(r){return '['+phase(r)+'] '+(r.vendor_folder?r.vendor_folder+' / ':'')+r.document_id;}
function fieldPath(field){
  if(!/^[a-z_]+(?:\[\d+\]|\.[a-z_]+)*$/.test(field))throw Error('Không hỗ trợ đường dẫn trường: '+field);
  return field.replace(/\[(\d+)\]/g,'.$1').split('.').map(x=>/^\d+$/.test(x)?Number(x):x);
}
function putField(data,path,value){
  let target=data;
  for(const part of path.slice(0,-1)){if(target?.[part]===undefined)throw Error('Dòng đã thay đổi; tải lại hóa đơn hoặc sửa JSON trực tiếp.');target=target[part];}
  target[path.at(-1)]=clone(value);
}
function applyFieldChoices(base,choices,diffs){
  const data=clone(base),removals=[],additions=[];
  for(const [index,side] of choices){
    const d=diffs[index],value=side==='A'?d.label_value:d.checker_value;
    if(d.field==='__evidence__')continue;
    const extra=/^(items|totals\.taxes)\[extra:(\d+)\]$/.exec(d.field);
    if(extra){if(side==='B')additions.push({path:fieldPath(extra[1]),index:Number(extra[2]),value:parseJSON(value)});continue;}
    const path=fieldPath(d.field);
    if(value==='<missing>'){removals.push(path);continue;}
    putField(data,path,parseJSON(value));
  }
  removals.sort((a,b)=>Number(b.at(-1))-Number(a.at(-1)));
  for(const path of removals){let parent=data;for(const part of path.slice(0,-1))parent=parent[part];if(!Array.isArray(parent))throw Error('Không thể bỏ khóa bắt buộc; sửa JSON trực tiếp.');parent.splice(path.at(-1),1);}
  additions.sort((a,b)=>a.index-b.index);
  for(const entry of additions){let rows=data;for(const part of entry.path)rows=rows[part];rows.splice(Math.min(entry.index,rows.length),0,clone(entry.value));}
  return data;
}
function chooseField(index,side){
  if(!current||busy)return;
  try{
    let state=fieldChoices.get(current.document_id);
    if(!state||state.text!==$('json').value)state={base:parseJSON($('json').value),choices:new Map()};
    const choices=new Map(state.choices);choices.set(index,side);
    const result=applyFieldChoices(state.base,choices,current.verification.differences||[]);
    state.choices=choices;$('json').value=JSON.stringify(result,null,2);state.text=$('json').value;
    fieldChoices.set(current.document_id,state);invalidate();renderVerification();
    message('Đã chọn model '+side+' cho trường này. Bấm Cập nhật để lưu và xác nhận reviewed.');
  }catch(e){message('Chưa chọn được: '+e.message,true);}
}
function renderVerification(){
  const v=current?.verification||{status:'pending'};
  $('verificationTitle').textContent=phase(current);
  $('verificationNote').textContent=v.error||('Bộ kiểm tra độc lập: '+(v.model||'chưa chạy')+(v.checked_at?' · '+v.checked_at:'')+'\nChọn A/B cho từng trường hoặc sửa JSON. Các lựa chọn chỉ được ghi khi bấm Cập nhật.');
  $('diffBody').replaceChildren();const diffs=v.differences||[];
  const state=fieldChoices.get(current.document_id),selected=state?.text===$('json').value?state.choices:new Map();
  diffs.forEach((d,index)=>{
    const tr=document.createElement('tr');
    for(const value of [d.field+(d.checker_field&&d.checker_field!==d.field?' → '+d.checker_field:''),d.label_value,d.checker_value]){const td=document.createElement('td');td.textContent=value;tr.append(td);}
    const td=document.createElement('td'),actions=document.createElement('div');actions.className='field-actions';
    for(const side of ['A','B']){const button=document.createElement('button');button.type='button';button.textContent='Chọn '+side;button.disabled=busy||d.field==='__evidence__';button.setAttribute('aria-pressed',String(selected.get(index)===side));button.onclick=()=>chooseField(index,side);actions.append(button);}
    td.append(actions);tr.append(td);$('diffBody').append(tr);
  });
  $('diffTable').hidden=!diffs.length;$('candidateDetails').hidden=!v.candidate;$('candidateJson').textContent=v.candidate?JSON.stringify(v.candidate,null,2):'';
}
function drawImage(){if(!current)return;const image=current.images[imageIndex],url='/'+image.file.split('/').map(encodeURIComponent).join('/');$('image').value=String(imageIndex);$('preview').src=url;$('original').href=url;$('preview').closest('.viewer').scrollTop=0;$('imagePrev').disabled=imageIndex===0;$('imageNext').disabled=imageIndex===current.images.length-1;$('name').textContent=current.source+' · ảnh '+(imageIndex+1)+'/'+current.images.length;}
function openDoc(id){current=records.find(r=>r.document_id===id)||null;$('main').hidden=!current;$('main').style.display=current?'grid':'none';$('empty').hidden=!!current;if(!current)return;imageIndex=0;$('image').replaceChildren();current.images.forEach((image,index)=>$('image').add(new Option(String(index+1),String(index))));$('json').value=current.editor_draft??JSON.stringify(current.annotation,null,2);$('json').scrollTop=0;$('dirty').textContent=current.editor_draft!==undefined?'Có bản sửa chưa ghi JSONL':'';message('');drawImage();badge();renderVerification();}
function list(prefer){const query=$('search').value.toLowerCase();visible=records.filter(r=>r.split===$('split').value&&JSON.stringify([r.document_id,r.source,r.vendor_folder]).toLowerCase().includes(query)&&matchesFilter(r,$('filter').value));$('doc').replaceChildren();visible.forEach(r=>$('doc').add(new Option(label(r),r.document_id)));const id=visible.some(r=>r.document_id===prefer)?prefer:visible[0]?.document_id;if(id)$('doc').value=id;openDoc(id);counts();}
function invalidate(){if(!current)return;current.editor_draft=$('json').value;current.review_status='needs_review';persist(current);$('dirty').textContent='Bản nháp · bấm Cập nhật để ghi JSONL';badge();}
function setBusy(value){busy=value;for(const id of ['json','update','unreview','useModel2','reload','doc','split','search','filter','prev','next','import','export'])$(id).disabled=value;for(const button of document.querySelectorAll('.field-actions button'))button.disabled=value||button.closest('tr').firstChild.textContent==='__evidence__';}
async function writeDisk(record){const response=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify(record)});const result=await response.json();if(!response.ok)throw Error(result.error||'Không ghi được JSONL');record._disk_revision=result.revision;if(result.verification)record.verification=result.verification;persist(record);return result;}
$('update').onclick=async()=>{if(!current||busy)return;const record=clone(current);try{record.annotation=parseJSON($('json').value);record.auto_labeled=autoLabeled(record);delete record.editor_draft;record.review_status='reviewed';record.review={...(record.review||{}),reviewed_at:new Date().toISOString(),action:'user_update_invoice'};setBusy(true);message('Đang ghi JSONL…');const result=await writeDisk(record);Object.assign(current,record);fieldChoices.delete(record.document_id);$('dirty').textContent='';list(record.document_id);message('Đã ghi annotations/'+result.file+'. Đã đồng bộ reviewed/'+result.file+'.');}catch(e){message('Chưa cập nhật: '+e.message,true);}finally{setBusy(false);}};
$('unreview').onclick=async()=>{if(!current||busy)return;const record=clone(current);try{record.auto_labeled=autoLabeled(record);record.review_status='needs_review';record.review={...(record.review||{}),reviewed_at:null,action:'user_unreview_invoice'};setBusy(true);message('\u0110ang c\u1eadp nh\u1eadt tr\u1ea1ng th\u00e1i...');const result=await writeDisk(record);Object.assign(current,record);list(record.document_id);message('\u0110\u00e3 chuy\u1ec3n v\u1ec1 ch\u1edd review; gi\u1eef nguy\u00ean nh\u00e3n v\u00e0 lo\u1ea1i kh\u1ecfi reviewed/'+result.file+'.');}catch(e){message('Ch\u01b0a c\u1eadp nh\u1eadt: '+e.message,true);}finally{setBusy(false);}};
$('useModel2').onclick=()=>{if(!current?.verification?.candidate||busy)return;fieldChoices.delete(current.document_id);$('json').value=JSON.stringify(current.verification.candidate,null,2);invalidate();renderVerification();message('Đã chép bộ kiểm tra độc lập vào bản sửa; đối chiếu ảnh rồi bấm Cập nhật.');};
$('reload').onclick=()=>{if(records.some(r=>r.editor_draft!==undefined)){message('Có bản sửa chưa lưu. Cập nhật hoặc xuất JSONL trước khi tải trạng thái mới.',true);return;}location.reload();};
$('json').addEventListener('input',()=>{if(current)fieldChoices.delete(current.document_id);invalidate();renderVerification();});
$('doc').onchange=()=>openDoc($('doc').value);
for(const id of ['split','filter','search'])$(id).addEventListener(id==='search'?'input':'change',()=>list());
for(const [id,step]of [['prev',-1],['next',1]])$(id).onclick=()=>{const index=visible.findIndex(r=>r.document_id===current?.document_id),next=visible[index+step];if(next){$('doc').value=next.document_id;openDoc(next.document_id);}};
$('image').onchange=()=>{imageIndex=Number($('image').value);drawImage();};
for(const [id,step]of [['imagePrev',-1],['imageNext',1]])$(id).onclick=()=>{imageIndex+=step;drawImage();};
$('preview').onclick=()=>window.open($('original').href,'_blank','noopener');
$('export').onclick=()=>{const rs=records.filter(r=>r.split===$('split').value).map(r=>{const result=clone(r);delete result._disk_revision;return result;});const url=URL.createObjectURL(new Blob([rs.map(r=>JSON.stringify(r)).join('\n')+'\n'],{type:'application/x-ndjson;charset=utf-8'}));const link=document.createElement('a');link.href=url;link.download=$('split').value+'.jsonl';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('import').onclick=()=>$('importFile').click();
$('importFile').onchange=async()=>{let saved=0;try{const file=$('importFile').files[0];if(!file)return;const incoming=(await file.text()).split(/\r?\n/).filter(x=>x.trim()).map(parseJSON);for(const r of incoming){const target=records.find(t=>t.document_id===r.document_id&&t.split===r.split);if(!target||r.label_format!=='invoice-v3'||r.source_sha256!==target.source_sha256||JSON.stringify(r.images)!==JSON.stringify(target.images))throw Error('JSONL phải khớp invoice-v3 và ảnh hiện tại.');}setBusy(true);for(const r of incoming){const target=records.find(t=>t.document_id===r.document_id&&t.split===r.split);r._disk_revision=target._disk_revision;await writeDisk(r);Object.assign(target,r);saved++;}list(current?.document_id);message('Đã nhập và ghi '+saved+' hóa đơn.');}catch(e){message('Đã ghi '+saved+' hóa đơn; dừng do: '+e.message,true);}finally{setBusy(false);$('importFile').value='';}};
(async()=>{try{const response=await fetch('/api/records');if(!response.ok)throw Error('Không đọc được ground truth');const data=parseJSON(await response.text());records=data.records;token=data.server_token;for(const r of records){try{const saved=parseJSON(localStorage.getItem(key(r))||'null');if(saved&&saved._disk_revision===r._disk_revision&&JSON.stringify(saved.images)===JSON.stringify(r.images))Object.assign(r,saved);}catch(e){}}list();}catch(e){$('counts').textContent='Chạy invoice-ocr label review rồi mở localhost. '+e.message;}})();
</script>
</body></html>
'''


def revision(record):
    return hashlib.sha256(dumps(record).encode('utf-8')).hexdigest()


class Store:
    def __init__(self, directory):
        self.directory = directory.resolve()
        self.lock = threading.Lock()
        self.token = secrets.token_urlsafe(32)
        if not (self.directory/"review.html").exists():
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory/"review.html").write_text(REVIEW_HTML, encoding="utf-8")

    @contextmanager
    def disk_lock(self):
        # Serialize read-modify-write across the browser server and model scripts.
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory/'.review.lock').open('a+b') as stream:
            if stream.seek(0, 2) == 0:
                stream.write(b'0')
                stream.flush()
            if os.name == 'nt':
                import msvcrt
                deadline = time.monotonic() + 15
                while True:
                    stream.seek(0)
                    try:
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Review dataset is busy; retry shortly')
                        time.sleep(0.05)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == 'nt':
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def reset(self, splits):
        with self.lock, self.disk_lock():
            changed = {}
            # Validate before modifying any split.
            datasets = {split:self.read(split) for split in splits}
            for records in datasets.values():
                for record in records:
                    validate_invoice(record['annotation'])
            for split, records in datasets.items():
                for record in records:
                    reset_for_verification(record)
                write_jsonl(self.directory/'annotations'/(split+'.jsonl'), records)
                write_jsonl(self.directory/'reviewed'/(split+'.jsonl'), [])
                changed[split] = len(records)
            return changed

    def read(self, split):
        if split not in ('train','valid','test'):
            raise ValueError('Invalid split')
        return read_jsonl(self.directory/'annotations'/(split+'.jsonl'))

    def dataset(self):
        with self.lock, self.disk_lock():
            records = []
            for split in ('train','valid','test'):
                for record in self.read(split):
                    record['_disk_revision'] = revision(record)
                    records.append(record)
        return {'records': records, 'server_token': self.token}

    def update(self, incoming):
        record = dict(incoming)
        for key in ('field_review','legacy_pages','migration_note','preparation'):
            record.pop(key, None)
        expected = record.pop('_disk_revision', None)
        with self.lock, self.disk_lock():
            records = self.read(record.get('split'))
            index = next((i for i,r in enumerate(records) if r['document_id']==record.get('document_id')), None)
            if index is None:
                raise ValueError('Unknown document')
            old = records[index]
            if expected != revision(old):
                raise ValueError('File đã thay đổi ở phiên khác. Xuất bản sửa, tải lại rồi nhập bản muốn giữ.')
            for field in ('document_id','source','source_sha256','split','images'):
                if record.get(field) != old.get(field):
                    raise ValueError('Document identity changed: '+field)
            if record.get('label_format') != 'invoice-v3':
                raise ValueError('Expected invoice-v3: one annotation per invoice')
            validate_invoice(record['annotation'])
            if record.get('review_status') not in ('reviewed','needs_review'):
                raise ValueError('Invalid review status')
            if record['review_status']=='reviewed' and 'editor_draft' in record:
                raise ValueError('Incomplete draft cannot be reviewed')
            record['images_sha256'] = images_digest(record['images'])
            refresh_verification(record)
            records[index] = record
            selected = reviewed_records(records)
            write_jsonl(self.directory/'annotations'/(record['split']+'.jsonl'), records)
            write_jsonl(self.directory/'reviewed'/(record['split']+'.jsonl'), selected)
            return {'revision': revision(record), 'file': record['split']+'.jsonl', 'review_status': record['review_status'], 'verification': record.get('verification')}


def handler(store):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type='application/json; charset=utf-8'):
            if not isinstance(body, bytes):
                body = dumps(body).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path in ('/','/groundtruth/review.html'):
                return self.reply(200, (store.directory/'review.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/records':
                return self.reply(200, store.dataset())
            target = (ROOT/path.lstrip('/')).resolve()
            if target.is_relative_to((ROOT/'data').resolve()) and target.is_file() and target.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.tif','.tiff'):
                return self.reply(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
            self.reply(404, {'error':'Not found'})

        def do_POST(self):
            if self.path != '/api/review':
                return self.reply(404, {'error':'Not found'})
            if self.headers.get('X-Review-Token') != store.token:
                return self.reply(403, {'error':'Invalid review session'})
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 10_000_000:
                    raise ValueError('Invalid request size')
                self.reply(200, store.update(loads(self.rfile.read(length))))
            except (ValueError,KeyError,TypeError) as error:
                self.reply(400, {'error':str(error)})
            except OSError as error:
                self.reply(500, {'error':'Không ghi được JSONL: '+str(error)})
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groundtruth', type=Path, default=ROOT/'groundtruth')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer(('127.0.0.1',args.port), handler(Store(args.groundtruth)))
    print(f'Mở http://127.0.0.1:{args.port}/groundtruth/review.html', flush=True)
    print('Một hóa đơn = một nhãn. Cập nhật ghi trực tiếp annotations/<split>.jsonl.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
