#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include <fcntl.h>

#define MAX_LEN 512         // Maximum command line length
#define MAXARGS 10          // Maximum number of arguments
#define ARGLEN 30           // Maximum argument length
#define PROMPT "PUCITshell@/home/mustafa/:- " // Shell prompt

// Function prototypes
int execute(char *arglist[]);
char **tokenize(char *cmdline);
char *read_cmd(char *, FILE *);
int handle_redirection_and_pipes(char *cmdline);

int main() {
    char *cmdline;
    char *prompt = PROMPT;

    // Main loop to read and process commands
    while ((cmdline = read_cmd(prompt, stdin)) != NULL) {
        handle_redirection_and_pipes(cmdline); // Process redirection and pipes
        free(cmdline);
    }
    printf("\n");
    return 0;
}

// Executes a command using fork and execvp
int execute(char *arglist[]) {
    int status;
    pid_t cpid = fork();

    switch (cpid) {
        case -1:
            perror("fork() failed");
            exit(1);
        case 0: 
            execvp(arglist[0], arglist);
            perror("!...command not found...!");
            exit(1);
        default:
            waitpid(cpid, &status, 0);
            return 0;
    }
}

// Tokenizes the command line into arguments
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

    // Parse command line into arguments
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

// Reads command line input from user
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

// Processes redirection and pipes in command
int handle_redirection_and_pipes(char *cmdline) {
    char *commands[MAXARGS + 1];
    int cmd_count = 0;

    // Split command line by pipes
    char *token = strtok(cmdline, "|");
    while (token != NULL && cmd_count < MAXARGS) {
        commands[cmd_count++] = token;
        token = strtok(NULL, "|");
    }
    commands[cmd_count] = NULL;

    int prev_fd = -1; // File descriptor for input of the next command
    for (int i = 0; i < cmd_count; i++) {
        char *args[MAXARGS + 1];
        int arg_count = 0;
        int in_redirect = 0, out_redirect = 0;
        int fd_in, fd_out;

        // Parse each command for arguments and redirection
        token = strtok(commands[i], " ");
        while (token != NULL && arg_count < MAXARGS) {
            if (strcmp(token, "<") == 0) { // Input redirection
                in_redirect = 1;
                token = strtok(NULL, " ");
                fd_in = open(token, O_RDONLY);
                if (fd_in < 0) {
                    perror("Failed to open input file");
                    return -1;
                }
            } else if (strcmp(token, ">") == 0) { // Output redirection
                out_redirect = 1;
                token = strtok(NULL, " ");
                fd_out = open(token, O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (fd_out < 0) {
                    perror("Failed to open output file");
                    return -1;
                }
            } else {
                args[arg_count++] = token;
            }
            token = strtok(NULL, " ");
        }
        args[arg_count] = NULL;

        int pipefd[2];
        if (i < cmd_count - 1) {
            if (pipe(pipefd) == -1) { // Create pipe for output
                perror("Failed to create pipe");
                return -1;
            }
        }

        pid_t pid = fork();
        if (pid == -1) { // Fork failure
            perror("Fork failed");
            return -1;
        }

        if (pid == 0) { // Child process for command execution
            if (in_redirect) {
                dup2(fd_in, STDIN_FILENO); // Redirect input
                close(fd_in);
            }
            if (out_redirect) {
                dup2(fd_out, STDOUT_FILENO); // Redirect output
                close(fd_out);
            }
            if (prev_fd != -1) { // Set up input from previous command
                dup2(prev_fd, STDIN_FILENO);
                close(prev_fd);
            }
            if (i < cmd_count - 1) { // Set up output for next command
                close(pipefd[0]);
                dup2(pipefd[1], STDOUT_FILENO);
                close(pipefd[1]);
            }

            execvp(args[0], args);
            perror("Exec failed");
            exit(1);
        } else { // Parent process to manage pipes and wait
            if (prev_fd != -1) {
                close(prev_fd);
            }
            if (i < cmd_count - 1) {
                close(pipefd[1]);
                prev_fd = pipefd[0];
            }
            waitpid(pid, NULL, 0); // Wait for child process to finish
        }
    }
    return 0;
}
