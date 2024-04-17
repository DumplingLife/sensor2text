from evaluation.caption_metrics import calculate_metrics_new

# UniVL
reworded_captions_updated = {
    "clean a pan with a sponge\n": "A person is cleaning a pan with a sponge.",
    "clean a pan with a towel\n": "A person is cleaning a pan with a towel.",
    "clean a plate with a sponge\n": "A person is cleaning a plate with a sponge.",
    "clean a plate with a towel\n": "A person is cleaning a plate with a towel.",
    "clear cutting board\n": "A person is clearing the cutting board.",
    "get / replace items from refrigerator / cabinets / drawers\n": "A person is replacing items in the refrigerator, cabinets, and drawers.",
    "get items from cabinets : 3 each large / small plates , bowls , mugs , glasses , sets of utensils\n": "A person is getting plates, bowls, mugs, glasses, and utensils from the cabinets.",
    "get items from refrigerator / cabinets / drawers\n": "A person is retrieving items from the refrigerator, cabinets, and drawers.",
    "load dishwasher : 3 each large / small plates , bowls , mugs , glasses , sets of utensils\n": "A person is loading the dishwasher with plates, bowls, mugs, glasses, and utensils.",
    "open / close a jar of almond butter\n": "A person is opening and closing a jar of almond butter.",
    "peel a cucumber\n": "A person is peeling a cucumber.",
    "peel a potato\n": "A person is peeling a potato.",
    "pour water from a pitcher into a glass\n": "A person is pouring water from a pitcher into a glass.",
    "set table : 3 each large / small plates , bowls , mugs , glasses , sets of utensils\n": "A person is setting the table with plates, bowls, mugs, glasses, and utensils.",
    "slice a cucumber\n": "A person is slicing a cucumber.",
    "slice a potato\n": "A person is slicing a potato.",
    "slice bread\n": "A person is slicing the bread.",
    "spread almond butter on a bread slice\n": "A person is spreading almond butter on a bread slice.",
    "spread jelly on a bread slice\n": "A person is spreading jelly on a bread slice.",
    "stack on table : 3 each large / small plates , bowls\n": "A person is putting plates and bowls onto the table.",
    "stack on table : 3 each large / small plates , bowls , mugs , glasses , sets of utensils\n": "A person is setting the table with plates, bowls, mugs, glasses, and utensils.",
    "unload dishwasher : 3 each large / small plates , bowls , mugs , glasses , sets of utensils\n": "A person is unloading the plates, bowls, mugs, glasses, and utensils from the dishwasher."
}

with open("evaluation/temp_ref.txt", "r") as file:
    ground_truth_captions = [reworded_captions_updated[caption] for caption in file]

with open("evaluation/temp_hyp.txt", "r") as file:
    generated_captions = [reworded_captions_updated[caption] for caption in file]

calculate_metrics_new(generated_captions, ground_truth_captions)