"""Seekable HTTP-range ZIP reader; refuses servers ignoring Range."""
import io
import urllib.request

class RemoteZipFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos, self.transferred = url, size, 0, 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos
    def read(self, n=-1):
        n = min(self.size-self.pos, n if n >= 0 else self.size-self.pos)
        if n <= 0: return b''
        start, end = self.pos, self.pos+n-1
        # Query distinguishes ranges in HTTP proxy caches.
        separator = '&' if '?' in self.url else '?'
        url = self.url + separator + f'range_start={start}&range_end={end}'
        req=urllib.request.Request(url, headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(req, timeout=90) as response:
            assert response.status == 206, f'Server ignored Range: {response.status}'
            assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
            data=response.read(n+1)
        assert len(data)==n, (len(data),n)
        self.pos += n
        self.transferred += n
        return data
