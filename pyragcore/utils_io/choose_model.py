import subprocess
import sys

from ragcore.exceptions import ModelNotFoundException
def choose_model()->str|None:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")[1:]
        models= [line.split()[0] for line in lines if line.strip()]
        print("\nAvailable models:")
        for i,model in enumerate(models,1):
            print(f"{i}. {model}")

        while True:
            choice = input("\nChoose a model number or press q to quit: ").strip()

            if choice.lower() == "q":
                sys.exit(0)

            if not choice.isdigit():
                print("Please enter a number.")
                continue
            if choice.isdigit() and 1<=int(choice)<=len(models):
                return models[int(choice)-1]
            print("Invalid choice.")
    except Exception as e:
        raise ModelNotFoundException(f"Failed to list models: {e}")
