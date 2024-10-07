import re
import matplotlib.pyplot as plt

def extract_accuracy_and_val_accuracy(line):
    # Updated regex to capture full scientific notation
    match = re.search(r"2116\/2116 \[==============================\] - [\d\.]+s [\d\.]+s\/step - loss: [\d\.]+ - accuracy: ([\d\.]+) - val_loss: [\d\.]+ - val_accuracy: ([\d\.]+)", line)
    if match:
        try:
            accuracy = float(match.group(1))
            val_accuracy = float(match.group(2))
            return accuracy, val_accuracy
        except ValueError:
            print(f"Skipping line due to value error: {line.strip()}")
            return None, None
    return None, None

def process_file(filename):
    accuracies = []
    val_accuracies = []
    
    with open(filename, 'r') as file:
        #for _ in range(skip_lines):
            #next(file)  # Skip the first 'skip_lines' lines
        
        for line in file:
            accuracy, val_accuracy = extract_accuracy_and_val_accuracy(line)
            if accuracy is not None and val_accuracy is not None:
                accuracies.append(accuracy)
                val_accuracies.append(val_accuracy)
            #else:
                #print(f"Skipping line: {line.strip()}")
    
    return accuracies, val_accuracies

def plot_accuracy_vs_iteration(iterations, accuracies):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(iterations) + 1)
    plt.plot(epochs, iterations, marker=',', linestyle='-', label='Accuracy')
    plt.plot(epochs, accuracies, marker=',', linestyle='-', label='Validation Accuracy')
    plt.title('Accuracy vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# Process the provided text file
filename = 'REU_Summer_2024\Output_Files\ResNet101V2.txt'  # Replace with your file path
accuracies, val_accuracies = process_file(filename)

# Plot the graph
plot_accuracy_vs_iteration(accuracies, val_accuracies)