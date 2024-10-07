import re
import matplotlib.pyplot as plt

def extract_accuracy_and_iteration(line):
    # Updated regex to capture full scientific notation
    match = re.search(r'^\s*(\d+)/2116\s+\[.*?\] - ETA: .*? - loss: .*? - accuracy: ([\d\.e\+\-]+)', line)
    if match:
        try:
            iteration = int(match.group(1))
            accuracy = float(match.group(2))
            return iteration, accuracy
        except ValueError:
            print(f"Skipping line due to value error: {line.strip()}")
            return None, None
    return None, None

def process_file(filename, skip_lines=12):
    iterations = []
    accuracies = []
    
    with open(filename, 'r') as file:
        for _ in range(skip_lines):
            next(file)  # Skip the first 'skip_lines' lines
        
        for line in file:
            iteration, accuracy = extract_accuracy_and_iteration(line)
            if iteration is not None and accuracy is not None:
                iterations.append(iteration)
                accuracies.append(accuracy)
            else:
                print(f"Skipping line: {line.strip()}")
    
    return iterations, accuracies

def plot_accuracy_vs_iteration(iterations, accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, accuracies, marker='.', linestyle='None')
    plt.title('Accuracy vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

# Process the provided text file
filename = 'REU_Summer_2024\Output_Files\Training.txt'  # Replace with your file path
iterations, accuracies = process_file(filename)

# Plot the graph
plot_accuracy_vs_iteration(iterations, accuracies)
