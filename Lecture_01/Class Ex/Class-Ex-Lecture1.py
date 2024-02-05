# =================================================================
# Class_Ex1:
# Write a function that prints all the chars from string1 that appears in string2.
# Note: Just use the strings functionality no other packages should be used.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')

def print_common_chars(string1, string2):
    common_chars = set(string1) & set(string2)

    if common_chars:
        print(f"Common characters in both strings: {' '.join(common_chars)}")
    else:
        print("No common characters found.")

string1 = "I am still"
string2 = "learning Python"
print_common_chars(string1, string2)

print(20 * '-' + 'End Q1' + 20 * '-')
# =================================================================
# Class_Ex2:
# Write a function that counts the numbers of a particular letter in a string.
# For example count the number of letter "a" in abstract.
# Note: Compare your function with a count method.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')

def count_letter(input_string, target_letter):
    count = 0
    for char in input_string:
        if char == target_letter:
            count += 1
    return count

# Example usage:
input_string = "abstract"
target_letter = "a"

# Using the custom count_letter function
result_custom = count_letter(input_string, target_letter)
print(f"Custom function count: {result_custom}")

# Using the count method
result_builtin = input_string.count(target_letter)
print(f"Builtin count method: {result_builtin}")

print(20 * '-' + 'End Q2' + 20 * '-')
# =================================================================
# Class_Ex3:
# Write a function that reads the Story text and finds the strings in the curly brackets.
# Note: You are allowed to use the strings methods
# Copy a text from wiki and add some curly bracket in the text call the string Story.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')

stringStory = 'The {Border Collie} is a British breed of herding dog of the collie type \
    of medium size. It originates in the region of the {Anglo-Scottish border}, and \
    descends from the {traditional sheepdogs} once found all over the {British Isles}. \
    It is kept mostly as a {working sheep-herding dog} or as a {companion animal}. \
    It competes with success in {sheepdog trials}. It has been claimed that it is \
    the most {intelligent} breed of dog (Source: Wikipedia).'

def find_curly_bracket_strings(stringStory):
    start_index = 0
    results = []

    while True:
        start_index = stringStory.find('{', start_index)
        if start_index == -1:
            break

        end_index = stringStory.find('}', start_index + 1)
        if end_index == -1:
            break

        result = stringStory[start_index + 1:end_index]
        results.append(result)

        start_index = end_index + 1

    return results

found_strings = find_curly_bracket_strings(stringStory)
print("Strings in curly brackets:", found_strings)

print(20 * '-' + 'End Q3' + 20 * '-')
# =================================================================
# Class_Ex4:
# Write a function that read the first n lines of a file.
# Use test_1.txt as sample text.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

def read_first_n_lines(file_path, n):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [file.readline().strip() for _ in range(n)]
    return lines

file_path = "test_1.txt"
n = 2  # Change this to the desired number of lines

first_n_lines = read_first_n_lines(file_path, n)
for line in first_n_lines:
    print(line)

print(20 * '-' + 'End Q4' + 20 * '-')
# =================================================================
# Class_Ex5:
# Write a function that read a file line by line and store it into a list.
# Use test_1.txt as sample text.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')

def read_file_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

file_path = "test_1.txt"
lines_list = read_file_lines(file_path)

for line in lines_list:
    print(line)

print(20 * '-' + 'End Q5' + 20 * '-')
# =================================================================
# Class_Ex6:
# Write a function that read two text files and combine each line from first
# text file with the corresponding line in second text file.
# Use T1.txt and T2.txt as sample text.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')

def combine_lines(file1_path, file2_path):
    combined_lines = []

    with open(file1_path, 'r', encoding='utf-8') as file1, open(file2_path, 'r', encoding='utf-8') as file2:
        for line1, line2 in zip(file1, file2):
            combined_line = f"{line1.strip()} {line2.strip()}"
            combined_lines.append(combined_line)

    return combined_lines

file1_path = "T1.txt"
file2_path = "T2.txt"
combined_lines_list = combine_lines(file1_path, file2_path)

for combined_line in combined_lines_list:
    print(combined_line)

print(20 * '-' + 'End Q6' + 20 * '-')
# =================================================================
# Class_Ex7:
# Write a function that create a text file where all letters of English alphabet
# put together by number of letters on each line (use n as argument in the function).
# Sample output
# function(3)
# ABC
# DEF
# ...
# ...
# ...
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

def function(n):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for i in range(0, len(alphabet), n):
        print(alphabet[i:i + n])

n = 3
function(n)

print(20 * '-' + 'End Q7' + 20 * '-')
# =================================================================
# Class_Ex8:
# Write a function that reads a text file and count number of words.
# Note: USe test_1.txt as a sample text.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q8' + 20 * '-')

def count_words_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        words = text.split()
        word_count = len(words)
    return word_count

file_path = "test_1.txt"
result = count_words_in_file(file_path)
print(f"Number of words in the file: {result}")

print(20 * '-' + 'End Q8' + 20 * '-')
# =================================================================
# Class_Ex9:
# Write a script that go over elements and repeat it each as many times as its count.
# Sample Output = ['o' ,'o', 'o', 'g' ,'g', 'f']
# Use Collections
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q9' + 20 * '-')

from collections import Counter

cnt = Counter({'o': 3, 'g': 2, 'f': 1})
result = list(cnt.elements())

print("Sample Output =", result)

print(20 * '-' + 'End Q9' + 20 * '-')
# =================================================================
# Class_Ex10:
# Write a program that appends couple of integers to a list
# and then with certain index start the list over that index.
# Note: use deque
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q10' + 20 * '-')

from collections import deque

list = ["a","b","c"]
deq = deque(list)

deq.append(1)
deq.append(2)
print("Original Deque:", deq)

# Rotate the deque, starting from index 5
index_to_start_over = 4
deq.rotate(-index_to_start_over)
print("New Deque:", deq)

print(20 * '-' + 'End Q10' + 20 * '-')
# =================================================================
# Class_Ex11:
# Write a script using os command that finds only directories, files and all directories, files in a  path.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q11' + 20 * '-')

import os

def find_directories_files(path):
    try:
        directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        print("Directories:")
        print('\n'.join(directories))

        print("\nFiles:")
        print('\n'.join(files))

        print("\nAll Directories and Files:")
        all_entries = [entry for entry in os.listdir(path)]
        print('\n'.join(all_entries))
    
    except FileNotFoundError:
        print(f"The path '{path}' does not exist.")
    except PermissionError:
        print(f"Permission error while trying to access the path '{path}'. Make sure you have the required permissions.")

path_to_search = "C:\\Users\Bailey"
find_directories_files(path_to_search)

print(20 * '-' + 'End Q11' + 20 * '-')
# =================================================================
# Class_Ex12:
# Write a script that create a file and write a specific text in it and rename the file name.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q12' + 20 * '-')

import os

original_file_name = "original_file.txt"
new_file_name = "renamed_file.txt"
text_to_write = "Hello, this is the content of the file!"

# Create the file and write the text
with open(original_file_name, 'w', encoding='utf-8') as file:
    file.write(text_to_write)

# Rename the file
try:
    os.rename(original_file_name, new_file_name)
    print(f"File '{original_file_name}' has been renamed to '{new_file_name}'.")
except FileNotFoundError:
    print(f"The file '{original_file_name}' does not exist.")
except FileExistsError:
    print(f"A file with the name '{new_file_name}' already exists. Please choose a different name.")
except PermissionError:
    print(f"Permission error while trying to rename the file.")

print(20 * '-' + 'End Q12' + 20 * '-')
# =================================================================
# Class_Ex13:
#  Write a script  that scan a specified directory find which is  file and which is a directory.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q13' + 20 * '-')

import os

def scan_directory(directory_path):
    try:
        entries = os.listdir(directory_path)
        for entry in entries:
            full_path = os.path.join(directory_path, entry)
            if os.path.isdir(full_path):
                print(f"{entry} is a directory.")
            elif os.path.isfile(full_path):
                print(f"{entry} is a file.")
            else:
                print(f"{entry} is neither a file nor a directory.")
    except FileNotFoundError:
        print(f"The directory '{directory_path}' does not exist.")
    except PermissionError:
        print(f"Permission error while trying to access the directory '{directory_path}'. Make sure you have the required permissions.")

directory_to_scan = "C:\\Users"
scan_directory(directory_to_scan)

print(20 * '-' + 'End Q13' + 20 * '-')

