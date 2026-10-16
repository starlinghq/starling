// malloc_trim(0) from another thread: does it release a live / exited thread arena's freed memory?
#include <malloc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static long rss_anon_kb(void) {
  FILE *f = fopen("/proc/self/status", "r");
  char line[256];
  long v = -1;
  while (fgets(line, sizeof line, f))
    if (strncmp(line, "RssAnon:", 8) == 0) v = atol(line + 8);
  fclose(f);
  return v;
}

static volatile int phase;

static void *worker(void *arg) {
  int n = *(int *)arg;
  void *p[64];
  for (int i = 0; i < n; i++) {
    p[i] = malloc(1 << 20);
    memset(p[i], 1, 1 << 20);
  }
  void *keep = malloc(64);  // a small live chunk above the freed range, so they cannot merge into top
  for (int i = 0; i < n; i++) free(p[i]);
  phase = 1;
  while (phase != 2) usleep(1000);
  free(keep);
  return NULL;
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 20;
  void *big = malloc(16 << 20);
  memset(big, 1, 16 << 20);
  free(big);
  pthread_t t;
  pthread_create(&t, NULL, worker, &n);
  while (phase != 1) usleep(1000);
  printf("thread alive, freed: RssAnon %ld kB\n", rss_anon_kb());
  malloc_trim(0);
  printf("after malloc_trim(0) from main, thread alive: RssAnon %ld kB\n", rss_anon_kb());
  phase = 2;
  pthread_join(t, NULL);
  printf("after join: RssAnon %ld kB\n", rss_anon_kb());
  malloc_trim(0);
  printf("after malloc_trim(0) from main, thread exited: RssAnon %ld kB\n", rss_anon_kb());
  return 0;
}
