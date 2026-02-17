#Online Shopping System in whichpayment is parent class and pay is the common mehtod
class payment:
    def pay(self, amount):
        print("Payment of Amount is being processed")
#credit card inherits from payment
class CreditCardPayment(payment):
    def pay(self,amount):
        print(f"processing credit card Payment of{amount}")
class paypalPayPayment(payment):
    def pay(self,amount):
        print(f"Processing Paypal payment of {amount}")
class BankTransferPayment(payment):
    def pay(self,amount):
        print(f"Proessing BankTransfer Payment of {amount}")
class ShoppingCart(payment):
    def pay(self,amount):
        print(f"Processing Shopping Cart payment of {amount}")
#Creating Objects for each payment mehtod
credit=CreditCardPayment()
paypal=paypalPayPayment()
bank=BankTransferPayment()
cart=ShoppingCart()
payments=[
    CreditCardPayment(),
    paypalPayPayment(),
    BankTransferPayment(),
    ShoppingCart()
]
#processing Payments
for payment_method in payments:
     payment_method.pay(100)
paypal.pay(200)
bank.pay(300)
cart.pay(400)
