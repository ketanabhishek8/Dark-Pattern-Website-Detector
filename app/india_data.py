"""Indian e-commerce style snippets for domain adaptation.

The original dataset comes from US/UK shopping sites, so the model had never seen rupee prices,
Indian delivery offers or local phrasing, and flagged them as dark patterns. TRAIN is generated from
templates and added to the training set only. HOLDOUT is written separately with different phrasing
and is used only for evaluation.
"""
import random

rng = random.Random(7)
PRODUCTS = ["Cotton Saree", "Silk Kurta Set", "Leather Sandals", "Steel Water Bottle", "Wireless Earbuds",
            "Jute Tote Bag", "Printed Bedsheet", "Pressure Cooker 5L", "Running Shoes", "Chikankari Top",
            "Brass Diya Set", "Denim Jacket", "Ceramic Mug", "Yoga Mat", "Smart Watch", "Basmati Rice 5kg",
            "Face Wash 100ml", "Kids School Bag", "Table Lamp", "Rain Jacket"]
PRICES = [149, 199, 249, 299, 349, 399, 449, 499, 599, 699, 799, 899, 999, 1199, 1299, 1499, 1799, 1999, 2499, 3499]


def price():
    p = rng.choice(PRICES)
    return rng.choice([f"₹{p:,}", f"Rs. {p:,}", f"Rs {p}", f"₹{p}", f"INR {p:,}"])


NORMAL_TEMPLATES = [
    lambda: f"{rng.choice(PRODUCTS)} {price()}",
    lambda: f"{price()} {price()} ({rng.choice([10, 20, 25, 30, 40, 50, 60])}% off)",
    lambda: f"MRP {price()} inclusive of all taxes",
    lambda: f"Free delivery on orders above {price()}",
    lambda: f"Free shipping on orders over {price()}",
    lambda: f"Delivery charges {price()} for orders below {price()}",
    lambda: f"{rng.choice([3.6, 3.9, 4.0, 4.1, 4.2, 4.4, 4.5, 4.7])} out of 5 based on {rng.randint(12, 4000):,} ratings",
    lambda: f"{rng.choice([3.8, 4.1, 4.3, 4.6])} stars | {rng.randint(20, 900)} reviews",
    lambda: f"{rng.randint(10, 800)} ratings and {rng.randint(5, 200)} reviews",
    lambda: f"Easy {rng.choice([7, 10, 14, 15, 30])} day returns and exchanges",
    lambda: f"{rng.choice([7, 10, 30])} days return policy",
    lambda: f"Delivery in {rng.randint(2, 7)}-{rng.randint(8, 10)} business days",
    lambda: f"Get it by {rng.choice(['Monday', 'Tuesday', 'Friday', 'Sunday'])}, {rng.randint(1, 28)} {rng.choice(['Oct', 'Nov', 'Dec', 'Jan'])}",
    lambda: f"Enter pincode to check delivery",
    lambda: f"Cash on delivery available",
    lambda: f"Pay with UPI, cards or net banking",
    lambda: f"No cost EMI from {price()} per month",
    lambda: f"Fabric: {rng.choice(['100% cotton', 'Pure silk', 'Linen blend', 'Rayon', 'Georgette'])}",
    lambda: f"Material: {rng.choice(['Stainless steel', 'Genuine leather', 'Ceramic', 'Solid wood', 'BPA free plastic'])}",
    lambda: f"Care: {rng.choice(['Machine wash', 'Hand wash cold', 'Dry clean only', 'Wipe with a dry cloth'])}",
    lambda: f"Fit: {rng.choice(['Regular', 'Slim', 'Relaxed', 'Straight'])}",
    lambda: f"Net quantity: {rng.choice(['1 unit', '2 pieces', '500 g', '1 kg', '750 ml'])}",
    lambda: f"Country of origin: India",
    lambda: f"Sold by {rng.choice(['RetailNet', 'Sharma Traders', 'Nova Retail', 'Craftwala'])}",
    lambda: f"Warranty: {rng.choice(['6 months', '1 year', '2 years'])} manufacturer warranty",
    lambda: f"© {rng.choice([2023, 2024, 2025, 2026])} {rng.choice(['ShopKart', 'Desi Bazaar', 'Craftwala', 'Nova Retail'])}. All rights reserved.",
    lambda: f"Customer care: Monday to Saturday, 9 am to 6 pm",
    lambda: f"Save {price()} on this order",
    lambda: f"Bank offer: {rng.choice([5, 10])}% instant discount on HDFC credit cards",
    lambda: f"Use code {rng.choice(['FESTIVE10', 'WELCOME', 'SAVE50'])} for {rng.choice([5, 10, 15])}% off",
    lambda: f"Colours may vary slightly due to screen settings",
    lambda: f"Size guide",
    lambda: f"Track your order",
    lambda: f"Similar products",
    lambda: f"Frequently bought together",
    lambda: f"{rng.choice(['Great quality', 'Good fit', 'Value for money', 'Nice colour'])}, {rng.choice(['happy with it', 'would buy again', 'as described', 'delivered on time'])}",
    # Listing-page text (discount labels, product names with specs, stock status, contact lines, navigation)
    lambda: f"{rng.choice(['Min.', 'Min', 'Up to', 'Upto', 'Flat', 'Extra'])} {rng.choice([10, 20, 30, 40, 50, 60, 70, 80])}% Off",
    lambda: f"{rng.choice([10, 20, 30, 40])}-{rng.choice([50, 60, 70, 80])}% off on {rng.choice(['kurtas', 'sneakers', 'mobiles', 'laptops', 'watches', 'home decor'])}",
    lambda: f"{rng.choice(['Redmi', 'Samsung Galaxy', 'Realme', 'Vivo', 'Moto', 'iQOO', 'Poco'])} {rng.choice(['A15', 'M34', 'Note 14', 'G85', 'Z9', 'X7'])} 5G ({rng.choice([64, 128, 256, 512])}GB, {rng.choice(['Black', 'Ocean Blue', 'Silver', 'Mint Green'])})",
    lambda: f"{rng.choice([0.8, 1, 1.5, 2])} Ton {rng.choice(['Split', 'Window', 'Inverter'])} AC",
    lambda: f"{rng.choice(['Havells', 'Bajaj', 'Prestige', 'Philips', 'Usha'])} {rng.choice(['1200W', '750W', '3 Litre', '15L', '5 Star'])} {rng.choice(['Mixer Grinder', 'Geyser', 'Iron', 'Air Fryer', 'Fan'])}",
    lambda: rng.choice(["In stock", "Currently unavailable", "Out of stock", "Available in 4 colours"]),
    lambda: f"{rng.choice(['Telephone', 'Phone', 'Call us', 'Helpline'])}: {rng.choice(['022', '011', '080', '1800'])}-{rng.randint(1000000, 9999999)}",
    lambda: f"{rng.choice([7, 10, 15, 30])} Days Easy Returns",
    lambda: f"{rng.choice(['Topwear', 'Footwear', 'Ethnic wear', 'Watches', 'Bags'])} for {rng.choice(['Men', 'Women', 'Kids'])}",
    lambda: rng.choice(["Top offers", "Best of electronics", "Beauty, food, toys & more", "Home & furniture", "Deals on fashion"]),
]

DARK_TEMPLATES = [
    ("Scarcity", lambda: f"Only {rng.randint(1, 5)} left at {price()}"),
    ("Scarcity", lambda: f"Hurry, only {rng.randint(1, 4)} left in stock"),
    ("Scarcity", lambda: f"Almost sold out! {rng.randint(2, 6)} pieces left"),
    ("Scarcity", lambda: f"Selling out fast in your pincode"),
    ("Urgency", lambda: f"Deal price {price()} ends in {rng.randint(0, 1):02d}:{rng.randint(10, 59)}:{rng.randint(10, 59)}"),
    ("Urgency", lambda: f"Price goes up to {price()} at midnight"),
    ("Urgency", lambda: f"Offer valid for the next {rng.randint(5, 30)} minutes only"),
    ("Urgency", lambda: f"Lightning deal: {rng.randint(60, 95)}% claimed"),
    ("Social Proof", lambda: f"{rng.choice(['Rohan', 'Ananya', 'Priyanka', 'Arjun', 'Neha'])} from {rng.choice(['Chennai', 'Jaipur', 'Kolkata', 'Indore', 'Nagpur'])} bought this {rng.randint(2, 50)} minutes ago"),
    ("Social Proof", lambda: f"{rng.randint(20, 300)} people are looking at this right now"),
    ("Social Proof", lambda: f"{rng.randint(500, 5000):,} bought in the last 24 hours"),
    ("Misdirection", lambda: f"No thanks, I don't want to save {price()}"),
    ("Misdirection", lambda: f"I'll pay more, no discount for me"),
]

N_NORMAL, N_DARK = 520, 150
TRAIN = ([(f(), 0, "Not Dark Pattern") for f in (rng.choice(NORMAL_TEMPLATES) for _ in range(N_NORMAL))] +
         [(f(), 1, c) for c, f in (rng.choice(DARK_TEMPLATES) for _ in range(N_DARK))])
TRAIN = list({t[0]: t for t in TRAIN}.values())  # drop duplicate texts

# Written separately with different wording from the templates; used only for evaluation.
HOLDOUT = [
    ("Special price ₹2,149", 0, None), ("Rs. 899 onwards", 0, None), ("Extra ₹150 off on prepaid orders", 0, None),
    ("Shipping is free above Rs 1,500", 0, None), ("Rated 4.4 by 1,892 customers", 0, None),
    ("Return within 10 days of delivery", 0, None), ("Ships from Bengaluru warehouse", 0, None),
    ("Pattern: Floral print", 0, None), ("Sleeve length: Three-quarter", 0, None),
    ("Pack of 3 handkerchiefs ₹299", 0, None), ("Seller rating 4.2 / 5", 0, None),
    ("Standard delivery by Thursday", 0, None), ("GST invoice available", 0, None),
    ("Questions and answers", 0, None), ("Wishlist", 0, None), ("Write a product review", 0, None),
    ("Replacement only for damaged items", 0, None), ("Offer price ₹549, you save ₹250", 0, None),
    ("Handcrafted by artisans in Kutch", 0, None), ("Dimensions: 30 x 20 x 10 cm", 0, None),
    ("Just 2 units remaining, order now", 1, "Scarcity"), ("Last few left in size M", 1, "Scarcity"),
    ("Grab it before it's gone! Only 1 left", 1, "Scarcity"),
    ("Sale ends tonight, don't miss ₹999 price", 1, "Urgency"), ("Timer: 09 min 42 sec left for this price", 1, "Urgency"),
    ("Book in the next 15 mins to lock this fare", 1, "Urgency"),
    ("Kavya in Lucknow just ordered this", 1, "Social Proof"), ("Trending: 64 people added this to cart today", 1, "Social Proof"),
    ("Bestseller! 3,400 orders this week", 1, "Social Proof"),
    ("No, I prefer paying the full ₹1,299", 1, "Misdirection"), ("Skip offer, I'm fine missing out", 1, "Misdirection"),
    ("Hurry, deal of the day ends soon", 1, "Urgency"),
]
