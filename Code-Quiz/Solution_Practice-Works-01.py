# SOLUTION QNO 1:

def calculate_bmi(weight, height):
    # Calculate BMI using the formula
    return weight / (height ** 2)

def interpret_bmi(bmi):
    # Provide an interpretation of the BMI
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# Prompt user for weight and height
weight = float(input("Enter your weight in kilograms: "))
height = float(input("Enter your height in meters: "))

# Calculate BMI
bmi = calculate_bmi(weight, height)

# Display the calculated BMI and its interpretation
print(f"Your BMI is: {bmi:.2f}")
print(f"Interpretation: {interpret_bmi(bmi)}")

#-------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
# SOLUTION QNO 2:

def count_vowels_and_consonants(input_string):
    vowels = "aeiouAEIOU"
    vowel_count = 0
    consonant_count = 0
    
    for char in input_string:
        if char.isalpha():  # Check if the character is a letter
            if char in vowels:
                vowel_count += 1
            else:
                consonant_count += 1
                
    return vowel_count, consonant_count

# Example usage:
input_string = "Hello, World!"
vowel_count, consonant_count = count_vowels_and_consonants(input_string)

print(f"Number of vowels: {vowel_count}")
print(f"Number of consonants: {consonant_count}")


#-------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
# SOLUTION QNO 3:

def count_char_frequency(input_string):
    # Create an empty dictionary to store the frequency of each character
    frequency_dict = {}
    
    # Iterate over each character in the input string
    for char in input_string:
        if char in frequency_dict:
            # Increment the count if the character is already in the dictionary
            frequency_dict[char] += 1
        else:
            # Add the character to the dictionary with a count of 1
            frequency_dict[char] = 1
            
    return frequency_dict

# Prompt the user to input a string
user_input = input("Enter a string: ")

# Get the frequency of each character in the input string
char_frequency = count_char_frequency(user_input)

# Print the result as a dictionary
print(char_frequency)


