#include <stdio.h>        // Standard I/O library for input/output functions
#include <stdlib.h>       // Standard library for memory allocation, process control, etc.
#include <unistd.h>       // POSIX library for various system calls (e.g., fork, exec)
#include <sys/types.h>    // Defines data types used in system calls
#include <sys/wait.h>     // For waiting for process termination
#include <fcntl.h>        // For file control operations (e.g., open)
#include <string.h>       // For string manipulation functions (e.g., strcpy, strcmp)
#include <signal.h>       // For signal handling functions (e.g., sigaction)
#include <readline/readline.h>  // For `readline()` function to read input from the terminal
#include <readline/history.h>   // For maintaining command history
#include <errno.h>        // For error number definitions and error handling
#include <limits.h>       // For system limits (e.g., PATH_MAX)

#define MAXARGS 10        // Maximum number of arguments for a command
#define ARGLEN 30         // Maximum length of each argument
#define PROMPT "MyShell"  // Custom shell prompt
#define HISTORY_FILE ".my_shell_history" // File for storing command history
#define MAX_JOBS 100      // Maximum number of background jobs to track

// Structure to hold background job information
typedef struct {
    pid_t pid;            // Process ID of the job
    char command[256];    // Command line that was run
} Job;

Job jobs[MAX_JOBS];       // Array to store background jobs
int job_count = 0;        // Number of active background jobs

// Function prototypes
int execute(char *arglist[], int input_fd, int output_fd, int error_fd, int background);
int handle_builtin(char *arglist[]);
char **tokenize(char *cmdline, int *background);
void handle_pipes_and_execute(char **arglist, int background);
void sigchld_handler(int signum);
void display_prompt(char *prompt);
void add_job(pid_t pid, const char *command);
void list_jobs();
void kill_job(pid_t pid);
int are_jobs_present(); // Function declaration for checking job presence

// Main function: the entry point of the shell program
int main() {
    struct sigaction sa;  // Struct to define the signal action
    sa.sa_handler = sigchld_handler;  // Set the handler for SIGCHLD
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP; // Restart syscalls, don't handle stopped child processes
    sigaction(SIGCHLD, &sa, NULL);    // Register the signal handler for child processes

    using_history();                 // Initialize command history
    read_history(HISTORY_FILE);      // Read previous history from the file

    char *cmdline;                   // Pointer to hold the command line input
    char **arglist;                  // Array of strings for arguments
    char prompt[PATH_MAX + 50];      // String to store the custom prompt

    while (1) {                      // Infinite loop for the shell to accept commands
        display_prompt(prompt);      // Display the shell prompt
        cmdline = readline(prompt);  // Read user input

        if (!cmdline) {              // Check for EOF (Ctrl+D)
            break;                   // Exit the loop if EOF is received
        }

        int background = 0;          // Flag to check if the command should run in the background
        if (cmdline[0] != '\0') {    // If the command line is not empty
            add_history(cmdline);    // Add command to history
            write_history(HISTORY_FILE); // Write the updated history to the file
        }

        // Handle command repetition using history (`!number` and `!!`)
        if (cmdline[0] == '!') {
            HIST_ENTRY *entry = NULL; // Pointer for history entry
            if (cmdline[1] == '!') {  // Handle `!!` (repeat last command)
                entry = previous_history(); // Get last command
            } else {
                int history_index = atoi(cmdline + 1) - 1; // Convert number after `!` to an index
                if (history_index >= 0 && history_index < history_length) {
                    entry = history_get(history_index + 1); // Get the specific history entry
                }
            }

            if (entry) {
                free(cmdline);       // Free the current command line
                cmdline = strdup(entry->line); // Duplicate the history entry line
                printf("Repeating command: %s\n", cmdline);
            } else {
                printf("No such command in history.\n"); // Error if history entry is invalid
                free(cmdline);
                continue;
            }
        }

        // Tokenize the input into arguments
        if ((arglist = tokenize(cmdline, &background)) != NULL) {
            if (handle_builtin(arglist) == 0) { // Check if the command is a built-in
                handle_pipes_and_execute(arglist, background); // Execute external commands with pipes
            }

            // Free allocated memory for arguments
            for (int j = 0; j < MAXARGS + 1; j++) {
                free(arglist[j]);
            }
            free(arglist);
        }
        free(cmdline); // Free the command line after processing
    }

    write_history(HISTORY_FILE); // Save history before exiting
    printf("\n");
    return 0;
}

// Function to display the custom shell prompt
void display_prompt(char *prompt) {
    char cwd[PATH_MAX]; // Buffer to store the current working directory
    if (getcwd(cwd, sizeof(cwd)) == NULL) { // Get the current directory
        perror("getcwd() error"); // Print error if `getcwd` fails
        strcpy(cwd, "unknown");   // Default to "unknown" if error occurs
    }
    snprintf(prompt, PATH_MAX + 50, "(%s)-[%s]-$ ", PROMPT, cwd); // Create the prompt format
}

// Function to handle built-in commands
int handle_builtin(char *arglist[]) {
    if (strcmp(arglist[0], "cd") == 0) { // Handle `cd` command
        if (arglist[1] == NULL) {
            fprintf(stderr, "cd: missing operand\n"); // Error if no directory provided
        } else if (chdir(arglist[1]) != 0) { // Change directory and check for failure
            perror("cd");
        }
        return 1; // Indicate it was a built-in command
    } else if (strcmp(arglist[0], "exit") == 0) { // Handle `exit` command
        printf("Exit Successfully!");
        exit(0); // Exit the shell
    } else if (strcmp(arglist[0], "jobs") == 0) { // Handle `jobs` command
        if (are_jobs_present()) { // Check if any jobs are present
            list_jobs(); // List active background jobs
            return 1;
        } else {
            printf("No jobs Found.\n"); // Print if no jobs are present
            return -1;
        }
    } else if (strcmp(arglist[0], "kill") == 0) { // Handle `kill` command
        if (arglist[1] == NULL) {
            fprintf(stderr, "kill: missing PID or job id\n");
        } else {
            int signal = SIGKILL; // Default signal is `SIGKILL`
            int pid;

            // Check for optional signal, e.g., `kill -9 <pid>`
            if (arglist[1][0] == '-') {
                signal = atoi(arglist[1] + 1); // Parse signal number
                pid = atoi(arglist[2]); // Parse PID
            } else {
                pid = atoi(arglist[1]); // Parse PID directly
            }

            if (pid <= 0 || kill(pid, signal) != 0) { // Check for valid PID and try to kill the process
                perror("kill"); // Print error if `kill` fails
            }
        }
        return 1;
    } else if (strcmp(arglist[0], "help") == 0) { // Handle `help` command
        printf("Built-in commands:\n");
        printf("  cd [directory] - change directory\n");
        printf("  exit - exit the shell\n");
        printf("  jobs - list background jobs\n");
        printf("  kill [-signal] <pid> - send a signal to a process\n");
        printf("  help - display this help message\n");
        return 1;
    }
    return 0; // Return 0 if the command is not a built-in
}

// Function to execute a command with redirection and piping
int execute(char *arglist[], int input_fd, int output_fd, int error_fd, int background) {
    pid_t cpid = fork(); // Fork a new process
    if (cpid == -1) {
        perror("fork() failed");
        exit(1); // Exit if fork fails
    }
    if (cpid == 0) { // Child process
        if (input_fd != -1) { // Redirect input if specified
            dup2(input_fd, STDIN_FILENO);
            close(input_fd);
        }
        if (output_fd != -1) { // Redirect output if specified
            dup2(output_fd, STDOUT_FILENO);
            close(output_fd);
        }
        if (error_fd != -1) { // Redirect error if specified
            dup2(error_fd, STDERR_FILENO);
            close(error_fd);
        }

        execvp(arglist[0], arglist); // Replace process with the command
        perror("execvp"); // Print error if exec fails
        exit(1); // Exit child process if exec fails
    } else { // Parent process
        if (!background) { // If the command is not in the background
            waitpid(cpid, NULL, 0); // Wait for the child process to complete
        } else {
            add_job(cpid, arglist[0]); // Add job to background jobs list
            printf("Background job started with PID %d\n", cpid);
        }
    }
    return 0;
}
