def check_password_validity(password):
    """
    Validates password based on the following criteria:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(char.isupper() for char in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(char.islower() for char in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(char.isdigit() for char in password):
        return False, "Password must contain at least one digit"
    
    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    if not any(char in special_chars for char in password):
        return False, "Password must contain at least one special character"
    
    return True, "Password is valid"


# Test the function
if __name__ == "__main__":
    test_password = input("Enter password to check: ")
    is_valid, message = check_password_validity(test_password)
    print(message)