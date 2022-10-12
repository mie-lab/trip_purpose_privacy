import os
import json
import googlemaps

with open("api_key.json", "r") as infile:
    api_key = json.load(infile)["api_key"]

out_path = "data"
RADIUS = 20  # meters

gmaps = googlemaps.Client(key=api_key)

dict_google_places = {}
# version 1: everything inside a radius
results = gmaps.places_nearby(location={"lat": lat, "lng": lon}, radius=RADIUS)
# # version 2: rank by distance, get everything
# interesting_type = ["restaurant", "store", "doctor", "gym", "park", "bar",
#                     "subway_station", "train_station", "university"]
# possibly_interesting = ["school", "cafe", "church", "dentist", "supermarket",
#                         "meal_takeaway", "night_club", "shopping_mall"]
# results = gmaps.places_nearby(location={"lat": lat, "lng": lon}, type=place_type, rankby="distance")

dict_google_places[(lat, lon)] = results

with open(os.path.join(out_path, "google_places.json"), "r") as outfile:
    json.dump(dict_google_places, outfile)
