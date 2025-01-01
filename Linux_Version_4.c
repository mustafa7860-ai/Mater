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
#define HISTORY_SIZE 10

// History array and count to store command history
char *history[HISTORY_SIZE];
int history_count = 0;

int execute(char *arglist[], int background);
char **tokenize(char *cmdline);
char *read_cmd(char *, FILE *);
void handle_sigchld(int signo);
void add_to_history(char *cmdline);
char *get_from_history(int index);

int main() {
    char *cmdline;
    char **arglist;
    char *prompt = PROMPT;

    // Set up SIGCHLD handler to clean up zombie processes
    struct sigaction sa;
    sa.sa_handler = handle_sigchld;
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    sigaction(SIGCHLD, &sa, NULL);

    // Main shell loop to read and execute commands
    while ((cmdline = read_cmd(prompt, stdin)) != NULL) {
        // Check if command is a history retrieval (e.g., "!1")
        if (cmdline[0] == '!') {
            int index = atoi(&cmdline[1]);
            if (index == -1) {
                index = history_count - 1;
            } else {
                index--;
            }
            if (index >= 0 && index < history_count) {
                cmdline = strdup(get_from_history(index));
                printf("%s\n", cmdline);
            } else {
                printf("No such command in history\n");
                free(cmdline);
                continue;
            }
        } else {
            add_to_history(cmdline);  // Add new command to history
        }

        int background = 0;
        int len = strlen(cmdline);
        if (cmdline[len - 1] == '&') {  // Check for background execution
            background = 1;
            cmdline[len - 1] = '\0';
        }

        // Tokenize and execute the command
        if ((arglist = tokenize(cmdline)) != NULL) {
            execute(arglist, background);
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

// Fork and execute the command, handling redirection and background processes
int execute(char *arglist[], int background) {
    int status;
    pid_t cpid = fork();

    if (cpid == -1) {
        perror("fork() failed");
        exit(1);
    } else if (cpid == 0) {
        // Handle I/O redirection if "<" or ">" found in arguments
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

        execvp(arglist[0], arglist);  // Execute command
        perror("!...command not found...!");
        exit(1);
    } else {
        if (background) {
            printf("[%d] %d\n", 1, cpid);  // Display background process ID
        } else {
            waitpid(cpid, &status, 0);  // Wait for child if not background
            printf("child exited with status %d \n", status >> 8);
        }
    }
    return 0;
}

// Tokenize input command line into an array of arguments
char **tokenize(char *cmdline) {
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

    arglist[argnum] = NULL;
    return arglist;
}

// Read command line input from the user
char *read_cmd(char *prompt, FILE *fp) {
    printf("%s", prompt);
    int c;
    int pos = 0;
    char *cmdline = (char *)malloc(sizeof(char) * MAX_LEN);

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

// Signal handler to reap child processes
void handle_sigchld(int signo) {
    while (waitpid(-1, NULL, WNOHANG) > 0);
}

// Add a command to history, maintaining the history size limit
void add_to_history(char *cmdline) {
    if (history_count < HISTORY_SIZE) {
        history[history_count++] = strdup(cmdline);
    } else {
        free(history[0]);
        for (int i = 1; i < HISTORY_SIZE; i++) {
            history[i - 1] = history[i];
        }
        history[HISTORY_SIZE - 1] = strdup(cmdline);
    }
}

// Retrieve a command from history by index
char *get_from_history(int index) {
    return history[index];
}
