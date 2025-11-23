import subprocess
import os
import sys

# Assume Multiwfn.exe is in the project root for now, or specify its new path
# You will need to update this path based on where Multiwfn.exe is actually located
MULTIWWFN_EXE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Multiwfn.exe') 
# If Multiwfn.exe is still within a binary folder, adjust like this:
# MULTIWFN_EXE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Multiwfn_3.8_dev_bin_Win64', 'Multiwfn.exe')

def run_multiwfn_and_pymultiwfn(input_file_path):
    """
    Runs Multiwfn and pymultiwfn on a given input file and captures their outputs.
    """
    multiwfn_output = None
    pymultiwfn_output = None

    # --- Run Multiwfn ---
    if not os.path.exists(MULTIWWFN_EXE_PATH):
        print(f"Error: Multiwfn executable not found at {MULTIWWFN_EXE_PATH}", file=sys.stderr)
        multiwfn_output = "ERROR: Multiwfn executable not found."
    else:
        try:
            # This is a placeholder command. You'll need to adapt it based on
            # how Multiwfn is invoked to produce a specific output (e.g., Function 0)
            # and what output file it generates or if it prints to stdout.
            # Example: Running Multiwfn with an input file and getting its output
            # For this example, let's assume we want to run Function 0 (Basic Info)
            # and it prints some summary to stdout. This often requires piping commands.
            # A common way to get non-interactive output is to provide an input script
            # or redirect input from a file.
            
            # For demonstration, let's just try to run it with the input file
            # and capture whatever it prints. This likely won't work for
            # specific function output without more specific commands.
            
            # The Multiwfn.exe from the provided folder structure does not output to stdout directly
            # for function results. It typically generates files.
            # For now, let's simulate running it and returning a placeholder output.
            
            # For a real scenario, you'd need to:
            # 1. Determine how to run Multiwfn non-interactively to get the desired output
            #    (e.g., via stdin redirection or a script file).
            # 2. Capture stdout/stderr or read a generated output file.
            
            print(f"Attempting to run Multiwfn with input: {input_file_path}")
            # Example placeholder for Multiwfn command
            # This will likely NEED adjustment based on actual Multiwfn usage.
            # A common pattern is to provide input via a pipe:
            # multiwfn_command = f'echo 0\n1\n | "{MULTIWWFN_EXE_PATH}" "{input_file_path}"'
            # Or for simple execution, if it just processes the file and exits:
            result = subprocess.run(
                [MULTIWWFN_EXE_PATH, input_file_path], # This is highly simplified
                capture_output=True, text=True, check=False, encoding='utf-8'
            )
            multiwfn_output = result.stdout + result.stderr
            if result.returncode != 0:
                print(f"Multiwfn exited with error code {result.returncode}", file=sys.stderr)
                
        except Exception as e:
            multiwfn_output = f"ERROR running Multiwfn: {e}"
            print(f"Error running Multiwfn: {e}", file=sys.stderr)

    # --- Run pymultiwfn ---
    try:
        print(f"Attempting to run pymultiwfn with input: {input_file_path}")
        # This is a placeholder for pymultiwfn execution.
        # You'll need to import and call the relevant pymultiwfn function.
        # Example:
        # from pymultiwfn.analysis import analyze_file
        # pymultiwfn_data = analyze_file(input_file_path, function_id=0) # Assuming an API like this
        # For demonstration, let's return a simple string representation
        pymultiwfn_output = f"Pymultiwfn simulated output for {input_file_path}"
    except ImportError:
        pymultiwfn_output = "ERROR: pymultiwfn not found or not installed correctly."
        print("Error: pymultiwfn not found or not installed correctly. Ensure it's in PYTHONPATH or installed in editable mode.", file=sys.stderr)
    except Exception as e:
        pymultiwfn_output = f"ERROR running pymultiwfn: {e}"
        print(f"Error running pymultiwfn: {e}", file=sys.stderr)

    return multiwfn_output, pymultiwfn_output

if __name__ == "__main__":
    # Example usage with a dummy input file
    # Ensure this path is correct after moving the examples folder
    example_dir = os.path.join(os.path.dirname(__file__), 'examples')
    test_file = os.path.join(example_dir, 'benzene.fch') # Choose a common example file

    if not os.path.exists(example_dir):
        print(f"Error: Examples directory not found at {example_dir}", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Running verification for: {test_file}")
    mw_out, pmw_out = run_multiwfn_and_pymultiwfn(test_file)

    print("\n--- Multiwfn Output ---")
    print(mw_out)

    print("\n--- Pymultiwfn Output ---")
    print(pmw_out)

    print("\n--- Comparison (Raw) ---")
    if mw_out == pmw_out:
        print("Raw outputs are identical (which is unlikely at this stage).")
    else:
        print("Raw outputs differ.")