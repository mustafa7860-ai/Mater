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
#define MAXVARS 100

struct var {
    char *str;   // Variable string
    int global;  // Flag for variable scope (global/local)
};

struct var variables[MAXVARS];
int var_count = 0;  // Count of defined variables

char *history[HISTORY_SIZE];  // Command history
int history_count = 0;         // Current history size
pid_t bg_processes[HISTORY_SIZE];  // Background processes
int bg_count = 0;              // Count of background processes

// Function prototypes
int execute(char *arglist[], int background);
char **tokenize(char *cmdline);
char *read_cmd(char *, FILE *);
void handle_sigchld(int signo);
void add_to_history(char *cmdline);
char *get_from_history(int index);
void built_in_commands(char *arglist[]);
void set_variable(char *name, char *value, int global);
char *get_variable_value(const char *name);
void list_variables();

int main() {
    char *cmdline;
    char **arglist;
    char *prompt = PROMPT;

    struct sigaction sa;
    sa.sa_handler = handle_sigchld;  // Handle child process termination
    sigaction(SIGCHLD, &sa, NULL);

    while ((cmdline = read_cmd(prompt, stdin)) != NULL) {
        // History command handling
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
            add_to_history(cmdline);  // Add command to history
        }

        int background = 0;
        int len = strlen(cmdline);
        if (cmdline[len - 1] == '&') {
            background = 1;  // Background process flag
            cmdline[len - 1] = '\0';
        }

        if ((arglist = tokenize(cmdline)) != NULL) {
            if (strcmp(arglist[0], "exit") == 0) {
                // Terminate all background processes and exit
                for (int i = 0; i < bg_count; i++) {
                    kill(bg_processes[i], SIGKILL);
                }
                for (int i = 0; i < history_count; i++) {
                    free(history[i]);
                }
                exit(0);
            } else {
                built_in_commands(arglist);  // Handle built-in commands
                execute(arglist, background);  // Execute external commands
            }

            for (int j = 0; j < MAXARGS + 1; j++) {
                free(arglist[j]);  // Free tokenized arguments
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
    pid_t cpid = fork();  // Create a child process

    if (cpid == -1) {
        perror("fork() failed");
        exit(1);
    } else if (cpid == 0) {
        // Child process
        for (int i = 0; arglist[i] != NULL; i++) {
            // Handle input and output redirection
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
        // Parent process
        if (background) {
            bg_processes[bg_count++] = cpid;  // Track background process
            printf("[%d] %d\n", bg_count, cpid);
        } else {
            waitpid(cpid, &status, 0);  // Wait for foreground process
        }
    }
    return 0;
}

char **tokenize(char *cmdline) {
    char **arglist = (char **)malloc(sizeof(char *) * (MAXARGS + 1));
    for (int j = 0; j < MAXARGS + 1; j++) {
        arglist[j] = (char *)malloc(sizeof(char) * ARGLEN);
    }

    if (cmdline[0] == '\0') {
        return NULL;  // No command entered
    }

    int argnum = 0;
    char *cp = cmdline;
    char *start;
    int len;

    while (*cp != '\0') {
        while (*cp == ' ' || *cp == '\t') {
            cp++;  // Skip whitespace
        }

        start = cp;
        len = 0;

        while (*cp != '\0' && *cp != ' ' && *cp != '\t') {
            len++;
            cp++;  // Get command argument
        }

        strncpy(arglist[argnum], start, len);
        arglist[argnum][len] = '\0';
        argnum++;

        if (argnum >= MAXARGS) {
            break;  // Limit reached
        }
    }

    arglist[argnum] = NULL;  // Null-terminate argument list
    return arglist;
}

char *read_cmd(char *prompt, FILE *fp) {
    printf("%s", prompt);
    int c;
    int pos = 0;
    char *cmdline = (char *)malloc(sizeof(char) * MAX_LEN);

    while ((c = getc(fp)) != EOF) {
        if (c == '\n') {
            break;  // End of command
        }
        cmdline[pos++] = c;
    }

    if (c == EOF && pos == 0) {
        free(cmdline);
        return NULL;  // EOF and no command
    }

    cmdline[pos] = '\0';  // Null-terminate command line
    return cmdline;
}

void handle_sigchld(int signo) {
    while (waitpid(-1, NULL, WNOHANG) > 0);  // Reap terminated child processes
}

void add_to_history(char *cmdline) {
    // Add command to history, managing size
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

char *get_from_history(int index) {
    return history[index];  // Retrieve command from history
}

void set_variable(char *name, char *value, int global) {
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%s=%s", name, value);

    // Update or add variable
    for (int i = 0; i < var_count; i++) {
        if (strncmp(variables[i].str, name, strlen(name)) == 0) {
            free(variables[i].str);
            variables[i].str = strdup(buffer);
            variables[i].global = global;
            return;
        }
    }
    if (var_count < MAXVARS) {
        variables[var_count].str = strdup(buffer);
        variables[var_count].global = global;
        var_count++;
    } else {
        fprintf(stderr, "Error: Variable storage limit reached.\n");
    }
}

char *get_variable_value(const char *name) {
    // Retrieve variable value by name
    for (int i = 0; i < var_count; i++) {
        if (strncmp(variables[i].str, name, strlen(name)) == 0) {
            return strchr(variables[i].str, '=') + 1;  // Return value after '='
        }
    }
    return NULL;  // Not found
}

void list_variables() {
    // List all defined variables
    for (int i = 0; i < var_count; i++) {
        printf("%s (%s)\n", variables[i].str, variables[i].global ? "global" : "local");
    }
}

void built_in_commands(char *arglist[]) {
    if (strcmp(arglist[0], "set") == 0 && arglist[1] != NULL && arglist[2] != NULL) {
        int global = (strcmp(arglist[3], "global") == 0) ? 1 : 0;
        set_variable(arglist[1], arglist[2], global);  // Set variable
    } else if (strcmp(arglist[0], "get") == 0 && arglist[1] != NULL) {
        char *value = get_variable_value(arglist[1]);  // Get variable value
        if (value) {
            printf("%s\n", value);
        } else {
            printf("Variable not found.\n");
        }
    } else if (strcmp(arglist[0], "listvars") == 0) {
        list_variables();  // List variables
    }
}
