# =============================================================================
# ENCAPSULATION EXAMPLE: Bank Account Management
# =============================================================================
# Encapsulation is the bundling of data (attributes) and methods that operate
# on that data into a single unit (class), while hiding the internal details
# from the outside world. This protects data integrity and provides controlled
# access through public methods (getters/setters)
# =============================================================================

class BankAccount:
    """
    A simple bank account class demonstrating encapsulation principles.
    - Private attributes (double underscore __) cannot be accessed directly
    - Public methods provide controlled access to private data
    """
    
    def __init__(self, account_holder, initial_balance):
        """
        Initialize the bank account with a holder name and initial balance.
        
        Args:
            account_holder (str): Name of the account holder
            initial_balance (float): Starting balance (must be positive)
        """
        self.__account_holder = account_holder  # Private attribute
        self.__balance = initial_balance        # Private attribute (encapsulated)
    
    # =========== GETTER METHODS - For reading private data safely ==========
    
    def get_balance(self):
        """Return the current account balance without allowing modification."""
        return self.__balance
    
    def get_account_holder(self):
        """Return the account holder's name."""
        return self.__account_holder
    
    # =========== PUBLIC METHODS - Business logic for account operations =====
    
    def deposit(self, amount):
        """
        Add money to the account.
        Encapsulation ensures valid amounts are deposited only.
        
        Args:
            amount (float): Amount to deposit
        """
        if amount <= 0:
            print("❌ Error: Deposit amount must be positive!")
        else:
            self.__balance += amount
            print(f"✓ Deposited ${amount:.2f}. New balance: ${self.__balance:.2f}")
    
    def withdraw(self, amount):
        """
        Remove money from the account with validation.
        Encapsulation protects balance from invalid operations.
        
        Args:
            amount (float): Amount to withdraw
        """
        if amount <= 0:
            print("❌ Error: Withdrawal amount must be positive!")
        elif amount > self.__balance:
            print(f"❌ Insufficient funds! Available balance: ${self.__balance:.2f}")
        else:
            self.__balance -= amount
            print(f"✓ Withdrew ${amount:.2f}. New balance: ${self.__balance:.2f}")
    
    def display_account_info(self):
        """Display complete account information in a formatted way."""
        print(f"\n--- Account Information ---")
        print(f"Account Holder: {self.__account_holder}")
        print(f"Balance: ${self.__balance:.2f}")
        print("---" * 8)


# ==================== TESTING THE ENCAPSULATION ========================

# Create a bank account object
account = BankAccount("John Doe", 1000)

# Use public methods to interact with private data
account.deposit(500)        # Add money
account.withdraw(200)       # Remove money
account.withdraw(2000)      # Try to withdraw more than balance (error handling)
account.display_account_info()  # View account details

# ⚠️  This would cause an error - cannot access private attributes directly:
# print(account.__balance)  # AttributeError: 'BankAccount' object has no attribute '__balance'

# ✓ Instead, use public getter methods:
print(f"\nFinal Balance: ${account.get_balance():.2f}")


