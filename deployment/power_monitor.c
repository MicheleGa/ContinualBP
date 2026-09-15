#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <signal.h>
#include <unistd.h>

#define INTERVAL_NS 50000000L // 50 ms; tune after benchmarking vcgencmd on your Pi
static volatile sig_atomic_t stop_flag = 0;
void handle_term(int sig) { stop_flag = 1; }

int main(int argc, char **argv) {
    const char *outfile = argc > 1 ? argv[1] : "power_log.csv";
    FILE *out = fopen(outfile, "w");
    setvbuf(out, NULL, _IOLBF, 0); // line-buffered so a kill -TERM doesn't lose the tail

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(3, &set);
    sched_setaffinity(0, sizeof(set), &set);

    signal(SIGTERM, handle_term);

    fprintf(out, "t_mono_s,rail_idx,current_A,voltage_V\n"); // long format

    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);

    while (!stop_flag) {
        FILE *p = popen("vcgencmd pmic_read_adc", "r");
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now); // stamp AFTER the call returns
        double t = now.tv_sec + now.tv_nsec / 1e9;

        if (p) {
            char line[128];
            double cur[26] = {0}, volt[26] = {0};
            int seen[26] = {0};
            while (fgets(line, sizeof(line), p)) {
                int idx; double val; char kind[8];
                char *lp = strchr(line, '(');
                if (lp && sscanf(lp, "(%d)=%lf", &idx, &val) == 2 && idx >= 0 && idx < 26) {
                    if (strstr(line, "current"))      { cur[idx] = val;  seen[idx] |= 1; }
                    else if (strstr(line, "volt"))     { volt[idx] = val; seen[idx] |= 2; }
                }
            }
            pclose(p);
            for (int i = 0; i < 26; i++)
                if (seen[i]) fprintf(out, "%.6f,%d,%.8f,%.8f\n", t, i, cur[i], volt[i]);
        }

        next.tv_nsec += INTERVAL_NS;
        while (next.tv_nsec >= 1000000000L) { next.tv_nsec -= 1000000000L; next.tv_sec++; }
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
    }
    fclose(out);
    return 0;
}