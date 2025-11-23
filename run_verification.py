import os
import sys
from consistency_verifier import ConsistencyVerifier

def main():
    # Configuration
    # Update this path to point to your compiled Multiwfn executable
    # On Windows, it might be "Multiwfn.exe" if in PATH, or a full path.
    # On Linux/WSL, it might be "./Multiwfn"
    multiwfn_bin = os.environ.get("MULTIWFN_BIN", "Multiwfn") 
    
    # Check if Multiwfn binary exists (optional, the verifier checks too)
    if not os.path.exists(multiwfn_bin) and not any(os.access(os.path.join(path, multiwfn_bin), os.X_OK) for path in os.environ["PATH"].split(os.pathsep)):
        print(f"Warning: Multiwfn binary '{multiwfn_bin}' not found. Please set MULTIWFN_BIN environment variable or edit this script.")

    verifier = ConsistencyVerifier(multiwfn_bin)

    # Define a test case
    # You need a sample file, e.g., 'examples/test.wfn'
    test_file = "examples/test.wfn" 
    
    if not os.path.exists(test_file):
        # Create a dummy test file if it doesn't exist for demonstration
        os.makedirs("examples", exist_ok=True)
        with open(test_file, "w") as f:
            f.write("Dummy WFN file content")
        print(f"Created dummy test file at {test_file}")

    # Commands to send to Multiwfn
    # Example: '1' (Show info), 'q' (Exit)
    commands = ["1", "q"]

    print(f"Verifying consistency for {test_file} with commands: {commands}")
    
    result = verifier.verify(test_file, commands)

    if result["match"]:
        print("SUCCESS: Outputs match exactly!")
    else:
        print("FAILURE: Outputs do not match.")
        print("--- Diff ---")
        print(result["diff"])
        
        # Save outputs for inspection
        with open("output_ref.txt", "w") as f:
            f.write(result["output_ref"])
        with open("output_py.txt", "w") as f:
            f.write(result["output_py"])
        print("Saved outputs to output_ref.txt and output_py.txt")

if __name__ == "__main__":
    main()
