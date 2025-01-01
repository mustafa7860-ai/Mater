#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include <signal.h>
#include <fcntl.h>

#define MAX_LEN 512
#define MAXARGS 10
#define ARGLEN 30
#define PROMPT "PUCITshell@/home/mustafa/:- "

int execute(char *arglist[], int background);
char **tokenize(char *cmdline);
char *read_cmd(char *, FILE *);
void handle_sigchld(int signo);

int main() {
    char *cmdline;
    char **arglist;
    char *prompt = PROMPT;

    // Set up signal handler to reap background processes
    struct sigaction sa;
    sa.sa_handler = handle_sigchld;
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    sigaction(SIGCHLD, &sa, NULL);

    // Main loop for reading and executing commands
    while ((cmdline = read_cmd(prompt, stdin)) != NULL) {
        int background = 0;
        int len = strlen(cmdline);

        // Check if command should run in background
        if (cmdline[len - 1] == '&') {
            background = 1;
            cmdline[len - 1] = '\0';
        }

        if ((arglist = tokenize(cmdline)) != NULL) {
            execute(arglist, background);

            // Free allocated memory for arguments
            for (int j = 0; j < MAXARGS + 1; j++) {
                free(arglist[j]);
            }
            free(arglist);
        }
        free(cmdline);
    }
    printf("\n");
    return 0;
}

int execute(char *arglist[], int background) {
    int status;
    pid_t cpid = fork();

    if (cpid == -1) {
        perror("fork() failed");
        exit(1);
    } else if (cpid == 0) {
        // Handle input/output redirection
        for (int i = 0; arglist[i] != NULL; i++) {
            if (strcmp(arglist[i], "<") == 0) {
                int fd = open(arglist[i + 1], O_RDONLY);
                if (fd == -1) {
                    perror("Failed to open input file");
                    exit(1);
                }
                dup2(fd, STDIN_FILENO);
                close(fd);
                arglist[i] = NULL;
            } else if (strcmp(arglist[i], ">") == 0) {
                int fd = open(arglist[i + 1], O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (fd == -1) {
                    perror("Failed to open output file");
                    exit(1);
                }
                dup2(fd, STDOUT_FILENO);
                close(fd);
                arglist[i] = NULL;
            }
        }

        // Execute command
        execvp(arglist[0], arglist);
        perror("!...command not found...!");
        exit(1);
    } else {
        if (background) {
            printf("[%d] %d\n", 1, cpid); // Print background process ID
        } else {
            waitpid(cpid, &status, 0);
            printf("child exited with status %d \n", status >> 8); // Wait for foreground process
        }
    }
    return 0;
}

char **tokenize(char *cmdline) {
    // Allocate memory for argument list
    char **arglist = (char **)malloc(sizeof(char *) * (MAXARGS + 1));
    for (int j = 0; j < MAXARGS + 1; j++) {
        arglist[j] = (char *)malloc(sizeof(char) * ARGLEN);
        memset(arglist[j], 0, ARGLEN);
    }

    if (cmdline[0] == '\0') {
        return NULL;
    }

    int argnum = 0;
    char *cp = cmdline;
    char *start;
    int len;

    // Tokenize command line input
    while (*cp != '\0') {
        while (*cp == ' ' || *cp == '\t') {
            cp++;
        }

        start = cp;
        len = 1;

        while (*++cp != '\0' && *cp != ' ' && *cp != '\t') {
            len++;
        }

        strncpy(arglist[argnum], start, len);
        arglist[argnum][len] = '\0';
        argnum++;

        if (argnum >= MAXARGS) {
            break;
        }
    }

    arglist[argnum] = NULL; // Null-terminate argument list
    return arglist;
}

char *read_cmd(char *prompt, FILE *fp) {
    printf("%s", prompt);
    int c;
    int pos = 0;
    char *cmdline = (char *)malloc(sizeof(char) * MAX_LEN);

    // Read command line input
    while ((c = getc(fp)) != EOF) {
        if (c == '\n') {
            break;
        }
        cmdline[pos++] = c;
    }

    if (c == EOF && pos == 0) {
        free(cmdline);
        return NULL;
    }

    cmdline[pos] = '\0';
    return cmdline;
}

// Signal handler for child process termination
void handle_sigchld(int signo) {
    while (waitpid(-1, NULL, WNOHANG) > 0);
}
