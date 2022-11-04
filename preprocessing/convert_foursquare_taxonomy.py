import os
import json

manual_entries = {
    "Home (private)": "Other",
    "General Entertainment": "Arts and Entertainment",
    "Drugstore / Pharmacy": "Health and Medicine",
    "Salon / Barbershop": "Business and Professional Services",
    "Government Building": "Arts and Entertainment",
    "Ferry": "Travel and Transportation",
    "Post Office": "Business and Professional Services",
    "Miscellaneous Shop": "Retail",
    "Ice Cream Shop": "Coffee and Dessert",
    "Pizza Place": "Dining",
    "Athletic & Sport": "Sports and Recreation",
    "Housing Development": "Other",
    "Food & Drink Shop": "Coffee and Dessert",
    "Gift Shop": "Retail",
    "Train Station": "Travel and Transportation",
    "Subway": "Travel and Transportation",
    "Cosmetics Shop": "Retail",
    "Library": "Education",
    "School": "Education",
    "Sandwich Place": "Dining",
    "Road": "Other",
    "General Travel": "Travel and Transportation",
    "Deli / Bodega": "Retail",
    "Neighborhood": "Other",
    "Gym / Fitness Center": "Sports and Recreation",
    "Residential Building (Apartment / Condo)": "Other",
    "Building": "Other",
    "Pet Store": "Retail",
    "Board Shop": "Retail",
    "Moving Target": "Other",
    "Cemetery": "Landmarks and Outdoors",
    "General College & University": "Education",
    "Flower Shop": "Retail",
    "Taco Place": "Dining",
    "Nightclub": "Nightlife",
    "Automotive Shop": "Retail",
    "Gas Station / Garage": "Travel and Transportation",
    "Paper / Office Supplies Store": "Retail",
    "Bike Shop": "Retail",
    "Sporting Goods Shop": "Retail",
    "Pool": "Sports and Recreation",
    "Spa / Massage": "Business and Professional Services",
    "Ramen /  Noodle House": "Dining",
    "Malaysian Restaurant": "Dining",
    "Ski Area": "Sports and Recreation",
    "University": "Education",
    "Professional & Other Places": "Business and Professional Services",
    "Bike Rental / Bike Share": "Business and Professional Services",
    "Record Shop": "Retail",
    "Salad Place": "Dining",
    "Military Base": "Other",
    "Video Game Store": "Retail",
    "Taxi": "Travel and Transportation",
    "Embassy / Consulate": "Other",
    "Other Nightlife": "Nightlife",
    "Accessories Store": "Retail",
    "Boat or Ferry": "Travel and Transportation",
    "Train": "Travel and Transportation",
    "Athletics & Sports": "Sports and Recreation",
    "Gym Pool": "Sports and Recreation",
    "Optical Shop": "Retail",
    "Basketball Stadium": "Arts and Entertainment",
    "Baseball Stadium": "Arts and Entertainment",
    "Field": "Landmarks and Outdoors",
    "Gourmet Shop": "Retail",
    "Dentist's Office": "Health and Medicine",
    "Plane": "Travel and Transportation",
    "Bus Line": "Travel and Transportation",
    "Jazz Club": "Arts and Entertainment",
    "Gym": "Sports and Recreation",
    "Grocery Store": "Retail",
    "Tech Startup": "Business and Professional Services",
    "Wine Shop": "Retail",
    "Soccer Stadium": "Arts and Entertainment",
    "Mall": "Retail",
    "Outdoors & Recreation": "Sports and Recreation",
    "Hobby Shop": "Retail",
    "Nightlife Spot": "Nightlife",
    "Furniture / Home Store": "Retail",
    "Shop & Service": "Retail",
    "Food": "Dining",
    "Soup Place": "Dining",
    "Arts & Crafts Store": "Retail",
    "Bridal Shop": "Retail",
    "Thrift / Vintage Store": "Retail",
    "Historic Site": "Arts and Entertainment",
    "Financial or Legal Service": "Business and Professional Services",
    "Smoke Shop": "Retail",
    "Light Rail": "Travel and Transportation",
    "Racetrack": "Sports and Recreation",
    "Mobile Phone Shop": "Retail",
    "Travel & Transport": "Travel and Transportation",
    "Car Wash": "Business and Professional Services",
    "Vegetarian / Vegan Restaurant": "Dining",
    "Burrito Place": "Dining",
    "Cafe": "Coffee and Dessert",
    "Antique Shop": "Retail",
    "Arts & Entertainment": "Arts and Entertainment",
    "Fish & Chips Shop": "Dining",
    "College & University": "Education",
    "Animal Shelter": "Business and Professional Services",
    "City": "Other",
    "Mac & Cheese Joint": "Dining",
    "Gluten-free Restaurant": "Dining",
    "Motorcycle Shop": "Retail",
    "Market": "Retail",
}

if __name__ == "__main__":
    with open(os.path.join("data", "foursquare_taxonomy_raw.json"), "r") as infile:
        tax = json.load(infile)

    converted_tax = {}
    for key in tax.keys():

        labels = tax[key]["full_label"]

        if "Foursquare" in labels[0]:
            continue

        # separate dining and drinking
        if labels[0] == "Dining and Drinking":
            if len(labels) > 1 and labels[1] == "Bar":
                converted_tax[labels[-1]] = "Nightlife"
            elif len(labels) > 1 and (
                labels[1] == "Cafes, Coffee, and Tea Houses"
                or labels[1] == "Dessert Shop"
                or labels[1] == "Juice Bar"
                or labels[1] == "Bakery"
            ):
                converted_tax[labels[-1]] = "Coffee and Dessert"
            else:
                converted_tax[labels[-1]] = "Dining"
        elif labels[0] == "Arts and Entertainment" and labels[-1] in [
            "Night Club",
            "Rock Club",
            "Strip Club",
            "Party Center",
        ]:
            converted_tax[labels[-1]] = "Nightlife"
        elif labels[0] == "Community and Government":
            if len(labels) > 1:
                if labels[1] == "Spiritual Center":
                    converted_tax[labels[-1]] = "Spiritual Center"
                elif labels[1] == "Education":
                    converted_tax[labels[-1]] = "Education"
        else:
            converted_tax[labels[-1]] = labels[0]

    # Add manual labels
    converted_tax.update(manual_entries)
    with open(os.path.join("data", "foursquare_taxonomy.json"), "w") as outfile:
        json.dump(converted_tax, outfile)

