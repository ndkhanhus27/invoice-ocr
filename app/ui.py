"""Gradio presentation layer."""
import logging
from pathlib import Path
from invoice_ocr.config import ROOT, parse_args
from invoice_ocr.storage.files import EXTENSIONS
from .service import UploadService

def build_ui(service):
    import gradio as gr

    def run(files):
        try:
            status, text, download, report = service.run(files)
            items = [] if not report else [[
                '' if item[key] is None else str(item[key])
                for key in ('item_code', 'description', 'quantity', 'unit_price', 'line_total')
            ] for item in report['extraction']['items']]
            return status, text, download, report, items
        except Exception as error:
            # Clear the previous document's result so it cannot be saved by mistake.
            logging.error('Invoice upload failed (%s)', type(error).__name__)
            message = str(error) if isinstance(error, ValueError) else f'Không thể trích xuất ({type(error).__name__}). Kiểm tra log server.'
            return message, '', None, None, []

    def save(report):
        try:
            return service.save(report)
        except Exception as error:
            # Do not expose connection strings or uploaded content in UI errors.
            return f'Chưa lưu được ({type(error).__name__}). Kiểm tra cấu hình DB và chạy db-init.'

    with gr.Blocks(title='Invoice OCR', analytics_enabled=False) as demo:
        backend = getattr(service.args, 'backend', 'transformers')
        model_name = service.args.ollama_model if backend == 'ollama' else service.args.model
        gr.Markdown('# Trích xuất hóa đơn\nUpload các ảnh của **một hóa đơn**, theo thứ tự trang. Lần đầu chạy model có thể mất nhiều thời gian.')
        gr.Textbox(label='Model đang dùng', value=f'{backend}: {model_name}', interactive=False)
        report = gr.State(None)
        files = gr.File(label='Ảnh hóa đơn (theo thứ tự trang)', file_count='multiple',
                        file_types=sorted(EXTENSIONS), type='filepath')
        run_button = gr.Button('Trích xuất', variant='primary')
        status = gr.Textbox(label='Kết quả', interactive=False)
        items = gr.Dataframe(headers=['Mã hàng', 'Mô tả', 'Số lượng', 'Đơn giá', 'Thành tiền'],
                             datatype=['str'] * 5, type='array', label='Danh sách hàng', interactive=False)
        result = gr.Code(label='Dữ liệu trích xuất', language='json', interactive=False)
        download = gr.File(label='Tải kết quả JSON', interactive=False)
        save_button = gr.Button('Lưu PostgreSQL', interactive=bool(service.args.database_url))
        db_status = gr.Textbox(label='Trạng thái lưu DB', interactive=False)
        if not service.args.database_url:
            gr.Markdown('Để bật lưu DB, đặt `INVOICE_DATABASE_URL` trong `.env` rồi chạy `python -m invoice_ocr db init`.')
        # Clear old results and prevent changing pages while an extraction is running.
        run_button.click(
            lambda: (None, '', '', None, [], '', gr.update(interactive=False),
                     gr.update(interactive=False), gr.update(interactive=False)),
            outputs=[report, status, result, download, items, db_status, files, run_button, save_button],
            queue=False).then(run, inputs=files, outputs=[status, result, download, report, items],
                              concurrency_limit=1).then(
            lambda: (gr.update(interactive=True), gr.update(interactive=True),
                     gr.update(interactive=bool(service.args.database_url))),
            outputs=[files, run_button, save_button])
        save_button.click(save, inputs=report, outputs=db_status)
        files.upload(lambda: ('', '', None, None, '', []),
                     outputs=[status, result, download, report, db_status, items], queue=False)
        files.clear(lambda: ('', '', None, None, '', []),
                    outputs=[status, result, download, report, db_status, items], queue=False)
    return demo


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    options = parser.parse_args(argv)
    args = parse_args(['--env-file', str(options.env_file), 'extract'])
    service = UploadService(args)
    build_ui(service).queue(max_size=8).launch(server_name=options.host, server_port=options.port,
                                             share=False, max_file_size='20mb')


if __name__ == '__main__':
    main()
