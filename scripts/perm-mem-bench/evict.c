// evict: drop files from the page cache (posix_fadvise DONTNEED), no root needed.
// usage: evict file [file ...]
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
int main(int argc, char** argv) {
  int rc = 0;
  for (int i = 1; i < argc; ++i) {
    int fd = open(argv[i], O_RDONLY);
    if (fd < 0) { perror(argv[i]); rc = 1; continue; }
    fdatasync(fd);
    if (posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) != 0) { perror("fadvise"); rc = 1; }
    close(fd);
  }
  return rc;
}
