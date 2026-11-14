#!/usr/bin/env python3
"""Local server for UMUD Annotator, including TIFF-to-PNG decoding."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from io import BytesIO
from pathlib import Path
import mimetypes

ROOT = Path(__file__).resolve().parent

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self):
        if self.path != '/decode-tiff':
            self.send_error(404, 'Unknown endpoint')
            return
        try:
            from PIL import Image, ImageOps
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 100 * 1024 * 1024:
                self.send_error(400, 'Invalid TIFF file size')
                return
            data = self.rfile.read(length)
            with Image.open(BytesIO(data)) as im:
                im.seek(0)  # use first frame if TIFF is multipage
                im = ImageOps.exif_transpose(im.copy())
                # Canvas displays RGB/RGBA consistently and PNG is lossless.
                if im.mode not in ('RGB', 'RGBA'):
                    if 'A' in im.getbands():
                        im = im.convert('RGBA')
                    else:
                        im = im.convert('RGB')
                out = BytesIO()
                im.save(out, format='PNG')
                payload = out.getvalue()
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)
        except ImportError:
            body = b'Pillow is not installed. Run: python -m pip install Pillow'
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = ('TIFF decode failed: ' + str(exc)).encode('utf-8', errors='replace')
            self.send_response(422)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

if __name__ == '__main__':
    import webbrowser
    from threading import Timer
    host, port = '127.0.0.1', 8000
    print(f'UMUD Annotator running at http://{host}:{port}')
    print('Keep this terminal open while using the app. Press Ctrl+C to stop.')
    Timer(0.8, lambda: webbrowser.open(f'http://{host}:{port}')).start()
    ThreadingHTTPServer((host, port), Handler).serve_forever()
