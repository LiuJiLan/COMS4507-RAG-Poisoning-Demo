import argparse
import json
import os
import re
import time
import requests
from collections import Counter
from urllib.parse import quote_plus


OUTPUT_FILE = "brisbane_corpus.json"

HEADERS = {
    "User-Agent": "BrisbaneCorpusStudentProject/1.0"
}

MIN_WORDS = 100
MAX_WORDS = 220

TARGETS = {
    "tourism": 80,
    "restaurant": 80,
    "university": 60,
    "transport": 35,
    "culture_food": 35,
}


WIKIPEDIA_SOURCES = [
    # Tourism
    {"title": "Brisbane", "topic": "tourism"},
    {"title": "South_Bank_Parklands", "topic": "tourism"},
    {"title": "Lone_Pine_Koala_Sanctuary", "topic": "tourism"},
    {"title": "Mount_Coot-tha,_Queensland", "topic": "tourism"},
    {"title": "Story_Bridge", "topic": "tourism"},
    {"title": "Roma_Street_Parkland", "topic": "tourism"},
    {"title": "City_Botanic_Gardens", "topic": "tourism"},
    {"title": "Queensland_Museum", "topic": "tourism"},
    {"title": "Gallery_of_Modern_Art,_Brisbane", "topic": "tourism"},
    {"title": "Brisbane_River", "topic": "tourism"},
    {"title": "New_Farm_Park", "topic": "tourism"},
    {"title": "Kangaroo_Point_Cliffs", "topic": "tourism"},
    {"title": "Brisbane_City_Hall", "topic": "tourism"},
    {"title": "Queen_Street_Mall", "topic": "tourism"},
    {"title": "Fortitude_Valley", "topic": "tourism"},
    {"title": "Howard_Smith_Wharves", "topic": "tourism"},
    {"title": "Queensland_Performing_Arts_Centre", "topic": "tourism"},
    {"title": "Museum_of_Brisbane", "topic": "tourism"},
    {"title": "Brisbane_Botanic_Gardens,_Mount_Coot-tha", "topic": "tourism"},
    {"title": "Wheel_of_Brisbane", "topic": "tourism"},
    {"title": "West_End,_Queensland", "topic": "tourism"},
    {"title": "Moreton_Bay", "topic": "tourism"},
    {"title": "Moreton_Island", "topic": "tourism"},
    {"title": "North_Stradbroke_Island", "topic": "tourism"},
    {"title": "Glass_House_Mountains", "topic": "tourism"},
    {"title": "Gold_Coast,_Queensland", "topic": "tourism"},
    {"title": "Sunshine_Coast,_Queensland", "topic": "tourism"},
    {"title": "Lamington_National_Park", "topic": "tourism"},
    {"title": "Springbrook_National_Park", "topic": "tourism"},

    # Transport (expanded to reach 35)
    {"title": "Transport_in_Brisbane", "topic": "transport"},
    {"title": "Translink_(Queensland)", "topic": "transport"},
    {"title": "Go_card", "topic": "transport"},
    {"title": "Queensland_Rail_City_network", "topic": "transport"},
    {"title": "Brisbane_Airport", "topic": "transport"},
    {"title": "CityCat", "topic": "transport"},
    {"title": "Brisbane_busways", "topic": "transport"},
    {"title": "South_East_Queensland_bus_rapid_transit", "topic": "transport"},
    {"title": "Brisbane_Metro", "topic": "transport"},
    {"title": "Gold_Coast_light_rail", "topic": "transport"},
    {"title": "Brisbane_cycle_network", "topic": "transport"},
    {"title": "Airtrain_(Queensland)", "topic": "transport"},
    # Supplement v3 additions
    {"title": "M1_Pacific_Motorway", "topic": "transport"},
    {"title": "M3_motorway,_Brisbane", "topic": "transport"},
    {"title": "Northern_Busway,_Brisbane", "topic": "transport"},
    {"title": "Eastern_Busway,_Brisbane", "topic": "transport"},
    {"title": "South_East_Busway", "topic": "transport"},
    {"title": "Eagle_Street_Pier_Ferry_Terminal", "topic": "transport"},
    {"title": "Cycling_in_Brisbane", "topic": "transport"},
    {"title": "Light_rail_in_Brisbane", "topic": "transport"},

    # Culture and food (culture_food topic, expanded to reach 35)
    {"title": "Culture_of_Australia", "topic": "culture_food"},
    {"title": "Queensland", "topic": "culture_food"},
    {"title": "Brisbane_Festival", "topic": "culture_food"},
    {"title": "Ekka", "topic": "culture_food"},
    {"title": "Australian_cuisine", "topic": "culture_food"},
    {"title": "Coffee_culture_in_Australia", "topic": "culture_food"},
    {"title": "Moreton_Bay_bug", "topic": "culture_food"},
    {"title": "Pavlova_(cake)", "topic": "culture_food"},
    {"title": "Anzac_biscuit", "topic": "culture_food"},
    {"title": "Tim_Tam", "topic": "culture_food"},
    {"title": "Vegemite", "topic": "culture_food"},
    {"title": "Australian_wine", "topic": "culture_food"},
    {"title": "Farmers_market", "topic": "culture_food"},
    # Supplement v3 additions
    {"title": "Brisbane_Lions", "topic": "culture_food"},
    {"title": "Riverfire", "topic": "culture_food"},
    {"title": "Brisbane_Comedy_Festival", "topic": "culture_food"},
    {"title": "Queensland_Symphony_Orchestra", "topic": "culture_food"},

    # Suburbs reclassified as tourism
    {"title": "South_Brisbane,_Queensland", "topic": "tourism"},
    {"title": "Spring_Hill,_Queensland", "topic": "tourism"},
    {"title": "Kangaroo_Point,_Queensland", "topic": "tourism"},
    {"title": "New_Farm,_Queensland", "topic": "tourism"},
    {"title": "Paddington,_Queensland", "topic": "tourism"},
]


RESTAURANT_SOURCES = [
    ("Agnes", "Fortitude Valley", "modern Australian food"),
    ("Essa", "Fortitude Valley", "modern Australian food"),
    ("Gerard's Bistro", "Fortitude Valley", "Middle Eastern inspired food"),
    ("Hellenika Brisbane", "Fortitude Valley", "Greek food"),
    ("Greca", "Howard Smith Wharves", "Greek food"),
    ("Yoko Dining", "Howard Smith Wharves", "Japanese food"),
    ("Stanley", "Howard Smith Wharves", "Cantonese food"),
    ("Donna Tella", "Brisbane City", "Italian food"),
    ("Julius Pizzeria", "South Brisbane", "Italian pizza"),
    ("Beccofino", "Teneriffe", "Italian food"),
    ("Bianca", "Fortitude Valley", "Italian food"),
    ("sAme sAme", "Fortitude Valley", "Thai food"),
    ("Honto", "Fortitude Valley", "Japanese food"),
    ("SK Steak and Oyster", "Fortitude Valley", "steak and seafood"),
    ("Sushi Room", "Fortitude Valley", "Japanese sushi"),
    ("1889 Enoteca", "Woolloongabba", "Italian food"),
    ("Montrachet", "Bowen Hills", "French food"),
    ("Rogue Bistro", "Newstead", "modern Australian food"),
    ("Restaurant Dan Arnold", "Fortitude Valley", "fine dining"),
    ("OTTO Ristorante Brisbane", "South Bank", "Italian food"),
    ("Blackbird Bar and Grill", "Brisbane City", "steak and modern Australian food"),
    ("Moo Moo The Wine Bar and Grill", "Brisbane City", "steak and wine"),
    ("Walter's Steakhouse", "Brisbane City", "steakhouse"),
    ("Madame Wu", "Brisbane City", "Asian fusion food"),
    ("Alchemy Restaurant and Bar", "Brisbane City", "modern Australian food"),
    ("Patina at Customs House", "Brisbane City", "modern Australian food"),
    ("Massimo Restaurant and Bar", "Brisbane City", "Italian food"),
    ("Phoenix Brisbane", "Brisbane City", "Chinese food"),
    ("Rothwell's Bar and Grill", "Brisbane City", "grill and bistro food"),
    ("Longwang", "Brisbane City", "Asian food"),
    ("The Lex", "Brisbane City", "modern Australian food"),
    ("Fiume Rooftop Bar", "Fortitude Valley", "casual dining and drinks"),
    ("Tillerman", "Brisbane City", "seafood"),
    ("Goldfinch Restaurant", "Brisbane City", "modern Australian food"),
    ("The Croft House", "Brisbane City", "modern Australian food"),
    ("Lennons Restaurant and Bar", "Brisbane City", "modern Australian food"),
    ("Southside Restaurant", "South Brisbane", "Asian food"),
    ("Chu the Phat", "South Brisbane", "Asian fusion food"),
    ("Ciao Papi", "Howard Smith Wharves", "Italian food"),
    ("Felons Brewing Co", "Howard Smith Wharves", "brewery food"),
    ("Felons Barrel Hall", "Howard Smith Wharves", "brewery food"),
    ("Mr Percival's", "Howard Smith Wharves", "riverside bar food"),
    ("New Shanghai Queens Plaza", "Brisbane City", "Chinese food"),
    ("Fat Noodle Brisbane", "Brisbane City", "Asian noodle dishes"),
    ("The Pancake Manor", "Brisbane City", "pancakes and casual food"),
    ("Ahmet's Turkish Restaurant", "South Bank", "Turkish food"),
    ("The Charming Squire", "South Bank", "brewery food"),
    ("Ole Spanish Restaurant", "South Bank", "Spanish food"),
    ("Munich Brauhaus South Bank", "South Bank", "German food"),
    ("The Plough Inn", "South Bank", "pub food"),
    ("Harajuku Gyoza South Bank", "South Bank", "Japanese gyoza"),
    ("Maeve Wine", "South Brisbane", "wine bar food"),
    ("The Gunshop Bistrot", "West End", "breakfast and bistro food"),
    ("La Lupa", "West End", "Italian food"),
    ("Bird's Nest Yakitori West End", "West End", "Japanese yakitori"),
    ("Punjabi Palace", "South Brisbane", "Indian food"),
    ("Hello Please", "South Brisbane", "Vietnamese food"),
    ("Billykart West End", "West End", "casual Australian food"),
    ("Bar Alto", "New Farm", "Italian food"),
    ("New Farm Deli", "New Farm", "Italian deli food"),
    ("Vine Restaurant Bar", "New Farm", "Italian food"),
    ("The Balfour Kitchen", "New Farm", "modern Australian food"),
    ("Zero Fox", "Teneriffe", "Korean and Japanese inspired food"),
    ("Mizu Japanese Restaurant", "Teneriffe", "Japanese food"),
    ("The Moray Cafe", "New Farm", "cafe food"),
    ("Mrs Brown's Bar and Kitchen", "Newstead", "casual dining"),
    ("NOTA", "Paddington", "modern Australian food"),
    ("Hai Hai Ramen", "Paddington", "Japanese ramen"),
    ("Moga Izakaya and Sushi", "Rosalie", "Japanese food"),
    ("Kettle and Tin", "Paddington", "casual dining"),
    ("Naim", "Paddington", "Middle Eastern inspired food"),
    ("Remy's", "Paddington", "burgers and casual food"),
    ("Hope and Anchor", "Paddington", "pub food"),
    ("King Tea Chinese Restaurant", "Paddington", "Chinese food"),
    ("Sichuan Bang Bang", "Paddington", "Sichuan food"),
    ("Landmark Restaurant", "Sunnybank", "Chinese yum cha"),
    ("Little Red Dumpling", "Sunnybank", "Chinese dumplings"),
    ("Genkotsu Ramen", "Toowong", "Japanese ramen"),
    ("Sunny Seoul BBQ", "Sunnybank", "Korean barbecue"),
    ("KushiMaru", "Brisbane City", "Japanese food"),
    ("Eat Street Northshore", "Hamilton", "international street food"),
]


UQ_OFFICIAL_SOURCES = [
    ("UQ St Lucia Campus", "campus facilities and student life", "https://www.uq.edu.au/"),
    ("UQ Gatton Campus", "regional campus and agricultural education", "https://www.uq.edu.au/"),
    ("UQ Herston Campus", "health and medical education", "https://www.uq.edu.au/"),
    ("UQ Library Services", "study spaces and academic resources", "https://www.library.uq.edu.au/"),
    ("UQ Student Services", "student support and wellbeing", "https://my.uq.edu.au/student-support"),
    ("UQ International Student Support", "international student guidance", "https://study.uq.edu.au/"),
    ("UQ Accommodation Information", "student housing and accommodation options", "https://my.uq.edu.au/"),
    ("UQ Union and Student Life", "clubs, societies, and student representation", "https://www.uqu.com.au/"),
    ("UQ Student Central", "student administration and enquiries", "https://my.uq.edu.au/"),
    ("UQ Sport Facilities", "sport, fitness, and recreation", "https://uqsport.com.au/"),
    ("UQ Lakes and Green Spaces", "campus landscape and outdoor spaces", "https://www.uq.edu.au/"),
    ("UQ Great Court", "historic campus space and student gathering area", "https://www.uq.edu.au/"),
    ("UQ Art Museum", "art exhibitions and cultural learning", "https://art-museum.uq.edu.au/"),
    ("UQ Advanced Engineering Building", "engineering teaching and research facilities", "https://www.uq.edu.au/"),
    ("UQ Oral Health Centre", "dental and oral health education", "https://dentistry.uq.edu.au/"),
    ("UQ Pharmacy Australia Centre", "pharmacy teaching and research", "https://pharmacy.uq.edu.au/"),
    ("UQ Business School", "business education and research", "https://business.uq.edu.au/"),
    ("UQ School of EECS", "computer science and engineering education", "https://eecs.uq.edu.au/"),
    ("UQ Faculty of EAIT", "engineering, architecture, and information technology", "https://eait.uq.edu.au/"),
    ("UQ Faculty of Science", "science teaching and research", "https://science.uq.edu.au/"),
    ("UQ Faculty of Medicine", "medical education and health research", "https://medicine.uq.edu.au/"),
    ("UQ BEL Faculty", "business, economics, and law education", "https://bel.uq.edu.au/"),
    ("UQ HASS Faculty", "humanities, arts, and social sciences", "https://hass.uq.edu.au/"),
    ("UQ Institute for Molecular Bioscience", "biomedical and molecular research", "https://imb.uq.edu.au/"),
    ("UQ Queensland Brain Institute", "neuroscience and brain research", "https://qbi.uq.edu.au/"),
    ("UQ AIBN", "bioengineering and nanotechnology research", "https://aibn.uq.edu.au/"),
    ("UQ Sustainable Minerals Institute", "resources and sustainability research", "https://smi.uq.edu.au/"),
    ("UQ Global Change Institute", "sustainability and global change research", "https://gci.uq.edu.au/"),
    ("UQ Careers and Employability", "career planning and employability support", "https://employability.uq.edu.au/"),
    ("UQ my.UQ Student Portal", "student information and online services", "https://my.uq.edu.au/"),
    ("UQ Learn Online Platform", "online course learning support", "https://learn.uq.edu.au/"),
    ("UQ Maps and Campus Navigation", "campus navigation and location information", "https://maps.uq.edu.au/"),
    ("UQ Parking Information", "parking and campus transport rules", "https://my.uq.edu.au/"),
    ("UQ Public Transport Access", "bus, ferry, and commuting information", "https://my.uq.edu.au/"),
    ("UQ Orientation Program", "new student transition activities", "https://orientation.uq.edu.au/"),
    ("UQ Academic Calendar", "semester dates and academic deadlines", "https://my.uq.edu.au/"),
    ("UQ Enrolment and Timetabling", "course enrolment and class planning", "https://my.uq.edu.au/"),
    ("UQ Exams and Assessment", "assessment rules and exam information", "https://my.uq.edu.au/"),
    ("UQ Student IT Support", "technology help for students", "https://my.uq.edu.au/"),
    ("UQ Printing Services", "printing and scanning facilities", "https://my.uq.edu.au/"),
    ("UQ Research Support", "research resources and academic assistance", "https://research.uq.edu.au/"),
    ("UQ Graduate School", "postgraduate research support", "https://graduate-school.uq.edu.au/"),
    ("UQ Scholarships", "financial support and scholarship information", "https://scholarships.uq.edu.au/"),
    ("UQ Fees and Payments", "tuition fees and payment information", "https://my.uq.edu.au/"),
    ("UQ Health and Wellbeing", "health support and wellbeing resources", "https://my.uq.edu.au/"),
    ("UQ Counselling Services", "mental health and counselling support", "https://my.uq.edu.au/"),
    ("UQ Disability Support", "accessibility and disability services", "https://my.uq.edu.au/"),
    ("UQ Career Fairs", "employment events and industry networking", "https://employability.uq.edu.au/"),
    ("UQ Clubs and Societies", "student organisations and campus activities", "https://www.uqu.com.au/"),
    ("UQ Residential Colleges", "college accommodation and student communities", "https://www.uq.edu.au/"),
    ("UQ Safety and Security", "campus safety and emergency information", "https://my.uq.edu.au/"),
    ("UQ Food and Retail", "campus shops, cafes, and services", "https://www.uq.edu.au/"),
    ("UQ Sustainability", "environmental responsibility and campus sustainability", "https://sustainability.uq.edu.au/"),
    ("UQ Employability Award", "student employability development", "https://employability.uq.edu.au/"),
    ("UQ Exchange Program", "overseas exchange and study abroad", "https://employability.uq.edu.au/"),
    ("UQ Study Abroad Support", "international learning opportunities", "https://employability.uq.edu.au/"),
    ("UQ Postgraduate Support", "support for postgraduate coursework students", "https://my.uq.edu.au/"),
    ("UQ HDR Student Services", "support for higher degree research students", "https://graduate-school.uq.edu.au/"),
    ("UQ Graduation Information", "graduation ceremony and completion information", "https://my.uq.edu.au/"),
    ("UQ Student Learning Support", "academic skills, learning advice, and study support", "https://my.uq.edu.au/"),
]


def clean_text(text):
    text = re.sub(r"==.*?==", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ")
    return text.strip()


def format_title(title, topic):
    title = title.replace("_", " ")
    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"\s+", " ", title).strip()

    words = title.split()

    if len(words) < 5:
        if topic == "restaurant":
            title = f"{title} Restaurant in Brisbane Queensland"
        elif topic == "university":
            title = f"{title} at The University of Queensland"
        else:
            title = f"{title} in Brisbane Queensland"

    words = title.split()
    if len(words) > 15:
        title = " ".join(words[:15])

    return title


def split_into_chunks(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        chunk_words = words[start:start + max_words]

        if len(chunk_words) >= min_words:
            chunks.append(" ".join(chunk_words))

        start += max_words

    return chunks


# Four rotating padding sentences for restaurant docs — must match _RESTAURANT_TEMPLATES count
_RESTAURANT_EXTRAS = [
    " Dining options like this are useful for visitors because restaurant choices often depend on location, cuisine type, and convenience during a short stay in Brisbane.",
    " For travellers planning a Brisbane itinerary, knowing the cuisine style and neighbourhood of a venue helps narrow down meal choices around sightseeing and transport.",
    " Visitors and locals alike benefit from clear information about dining venues, as location, cuisine, and atmosphere all factor into choosing where to eat in Brisbane.",
    " Understanding what a venue offers in terms of cuisine and location helps travellers make efficient meal decisions while exploring different Brisbane precincts.",
]


def ensure_word_range(content, topic, idx=0):
    content = clean_text(content)

    if topic == "restaurant":
        extra = _RESTAURANT_EXTRAS[idx % len(_RESTAURANT_EXTRAS)]
    elif topic == "university":
        extra = (
            " This information is useful for students, visitors, and new arrivals because it explains how university "
            "services, campus facilities, and support resources are connected to everyday study life in Brisbane."
        )
    elif topic == "transport":
        extra = (
            " This information helps visitors understand how transport services connect major destinations, campuses, "
            "accommodation areas, and attractions across Brisbane."
        )
    elif topic == "culture_food":
        extra = (
            " This information helps visitors appreciate Brisbane's food culture, local culinary traditions, and the "
            "broader cultural context that shapes dining and lifestyle experiences in Queensland."
        )
    else:
        extra = (
            " This information helps visitors understand the local context, practical travel value, and relevance of "
            "this place when planning activities in Brisbane."
        )

    while len(content.split()) < MIN_WORDS:
        content += extra

    words = content.split()
    if len(words) > 300:
        content = " ".join(words[:MAX_WORDS])

    return content


def get_next_doc_id(docs):
    return f"brisbane_{len(docs) + 1:03d}"


def topic_count(docs, topic):
    return sum(1 for d in docs if d["topic"] == topic)


def can_add_topic(docs, topic):
    return topic_count(docs, topic) < TARGETS.get(topic, 999)


def fetch_wikipedia_extract(page_title):
    api_url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "titles": page_title,
        "redirects": 1
    }

    max_retries = 4

    for attempt in range(max_retries):
        try:
            response = requests.get(api_url, headers=HEADERS, params=params, timeout=20)

            if response.status_code == 429:
                wait_time = 10 + attempt * 10
                print(f"Rate limited: {page_title}. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                print(f"Failed: {page_title}, status code: {response.status_code}")
                return None, None

            data = response.json()
            pages = data.get("query", {}).get("pages", {})

            for _, page in pages.items():
                real_title = page.get("title", page_title.replace("_", " "))
                extract = page.get("extract", "")
                return real_title, clean_text(extract)

        except Exception as e:
            print(f"Error fetching {page_title}: {e}")
            time.sleep(5)

    return None, None


MAX_CHUNKS_PER_ARTICLE = 3


def generate_wikipedia_docs(docs):
    print("Generating Wikipedia documents...")

    # Group sources by topic so we can do round-robin across articles,
    # preventing any single article from consuming the whole topic quota.
    from collections import defaultdict
    by_topic = defaultdict(list)
    for source in WIKIPEDIA_SOURCES:
        by_topic[source["topic"]].append(source)

    # Fetch all articles first (title → (real_title, chunks)), capped per article.
    article_cache = {}
    all_sources_ordered = []
    for topic, sources in by_topic.items():
        for source in sources:
            all_sources_ordered.append(source)

    fetched_titles = []
    failed_titles = []

    # Fetch every article (respecting API politeness)
    for source in all_sources_ordered:
        page_title = source["title"]
        if page_title in article_cache:
            continue

        real_title, extract = fetch_wikipedia_extract(page_title)

        if not extract:
            print(f"Skipped: {page_title}, no extract")
            article_cache[page_title] = None
            failed_titles.append(page_title)
            time.sleep(1)
            continue

        all_chunks = split_into_chunks(extract)
        if not all_chunks:
            print(f"Skipped: {page_title}, content too short")
            article_cache[page_title] = None
            failed_titles.append(page_title)
            time.sleep(1)
            continue

        # Cap chunks per article to keep diversity high
        capped_chunks = all_chunks[:MAX_CHUNKS_PER_ARTICLE]
        article_cache[page_title] = (real_title, capped_chunks)
        fetched_titles.append(page_title)
        time.sleep(3)

    # Diagnostic: how many articles succeeded?
    print(f"Wikipedia: requested {len(all_sources_ordered)}, succeeded {len(fetched_titles)}, failed {len(failed_titles)}")
    if len(fetched_titles) < 30:
        print(f"⚠️  WARNING: Only {len(fetched_titles)} unique articles fetched (expected ≥ 30). Diversity may be low.")

    # Round-robin: for each topic, cycle through all its articles one chunk at
    # a time until the topic quota is full.  This prevents early articles from
    # monopolising the quota.
    for topic, sources in by_topic.items():
        # Build per-topic queue: list of (page_title, real_title, chunk_list)
        topic_articles = []
        for source in sources:
            cached = article_cache.get(source["title"])
            if cached is not None:
                real_title, chunks = cached
                topic_articles.append([source["title"], real_title, list(chunks)])

        if not topic_articles:
            print(f"No articles available for topic '{topic}'")
            continue

        # Rotate through articles, taking one chunk at a time, until quota full
        while can_add_topic(docs, topic):
            made_progress = False
            for article in topic_articles:
                if not can_add_topic(docs, topic):
                    break
                page_title, real_title, remaining_chunks = article
                if not remaining_chunks:
                    continue

                chunk = remaining_chunks.pop(0)
                made_progress = True

                # Build a descriptive sub-title based on the chunk's first sentence
                first_sentence = chunk.split(".")[0].strip()
                sub_words = first_sentence.split()[:6]
                sub_hint = " ".join(sub_words)
                num_used = MAX_CHUNKS_PER_ARTICLE - len(remaining_chunks)

                if num_used == 1 and len(article_cache[page_title][1]) == 1:
                    # Only one chunk from this article — use plain title
                    doc_title = real_title
                else:
                    doc_title = f"{real_title} — {sub_hint}"

                doc = {
                    "doc_id": get_next_doc_id(docs),
                    "title": format_title(doc_title, topic),
                    "content": ensure_word_range(chunk, topic),
                    "source": "wikipedia",
                    "topic": topic,
                    "url": f"https://en.wikipedia.org/wiki/{page_title}"
                }
                docs.append(doc)

            if not made_progress:
                # All articles exhausted for this topic
                break

    # Summary: unique URLs actually used
    from collections import Counter as _Counter
    url_counts = _Counter(d["url"] for d in docs if d["source"] == "wikipedia")
    print(f"Unique Wikipedia URLs in corpus: {len(url_counts)}")
    print(f"Top 5 by chunk count: {url_counts.most_common(5)}")
    print("Wikipedia documents completed.")


def make_tripadvisor_search_url(name, location):
    query = quote_plus(f"{name} {location} Brisbane")
    return f"https://www.tripadvisor.com.au/Search?q={query}"


# Four content templates for restaurant docs — rotated to avoid embedding collapse
_RESTAURANT_TEMPLATES = [
    # Template A: visitor recommendation tone
    (
        "{name} is a dining destination in the {location} area of Brisbane, known for its {cuisine}. "
        "Visitors to Brisbane often consider this venue when looking for a meal that reflects the city's "
        "diverse food scene. The restaurant is well-suited to travellers exploring nearby attractions, "
        "waterfront precincts, or inner-city neighbourhoods. For those planning a Brisbane itinerary, "
        "dining options such as this help balance sightseeing with local culinary experiences."
    ),
    # Template B: local reputation tone
    (
        "{name} has built a reputation in Brisbane's dining scene as a go-to spot for {cuisine}, "
        "drawing both locals and visitors to {location}. The venue reflects the broader mix of "
        "international and modern Australian food culture that characterises Brisbane's restaurant "
        "landscape. Guests looking for a meal in this part of the city will find it a practical "
        "choice given its location relative to transport links, hotels, and major attractions."
    ),
    # Template C: practical information tone
    (
        "Located in {location}, {name} serves {cuisine} and is a recognised part of Brisbane's "
        "hospitality offering. The restaurant is a common consideration for travellers who want to "
        "experience the city's food culture without venturing far from central areas. It sits "
        "alongside other dining, shopping, and entertainment options that make {location} a popular "
        "destination for both short-stay visitors and residents seeking variety in their dining choices."
    ),
    # Template D: comparative / standout tone
    (
        "Among Brisbane's many dining options, {name} in {location} is noted for its focus on "
        "{cuisine}. The venue contributes to the variety of food experiences available across the "
        "city's inner suburbs and waterfront precincts. Travellers who prioritise cuisine type when "
        "planning meals during a Brisbane visit will find this venue a relevant option alongside "
        "the area's cafés, bars, and other restaurants catering to a range of tastes and budgets."
    ),
]


def generate_restaurant_docs(docs):
    print("Generating restaurant documents...")

    for idx, (name, location, cuisine) in enumerate(RESTAURANT_SOURCES):
        if not can_add_topic(docs, "restaurant"):
            break

        # Rotate through the four templates
        template = _RESTAURANT_TEMPLATES[idx % len(_RESTAURANT_TEMPLATES)]
        content = template.format(name=name, location=location, cuisine=cuisine)

        # Vary the title suffix to avoid "X Restaurant in Brisbane" for every entry
        title_variants = [
            f"{name} Restaurant in Brisbane",
            f"{name} Dining in {location} Brisbane",
            f"{name} Brisbane {location}",
            f"{name} — {location} Brisbane",
        ]
        raw_title = title_variants[idx % len(title_variants)]

        doc = {
            "doc_id": get_next_doc_id(docs),
            "title": format_title(raw_title, "restaurant"),
            "content": ensure_word_range(content, "restaurant", idx=idx),
            "source": "local_curated",   # honest: synthesised from public knowledge
            "topic": "restaurant",
            "url": ""   # no live URL — avoids stale TripAdvisor search links
        }

        docs.append(doc)

    print(f"Restaurant documents added: {topic_count(docs, 'restaurant')}")


def generate_uq_docs(docs):
    print("Generating UQ university documents...")

    for title, focus, url in UQ_OFFICIAL_SOURCES:
        if not can_add_topic(docs, "university"):
            break

        content = (
            f"{title} is part of The University of Queensland's student, teaching, research, and campus support "
            f"network. It is primarily concerned with {focus}. UQ operates across multiple campuses in Brisbane, "
            f"with its main campus at St Lucia on the Brisbane River. Students and visitors can access relevant "
            f"information through the my.UQ student portal or by contacting UQ Student Central directly. "
            f"Whether navigating academic requirements, locating campus facilities, or seeking personal support, "
            f"UQ provides a range of services designed to assist students throughout their studies in Brisbane."
        )

        # Vary title suffix to avoid every UQ doc ending "at The University of Queensland"
        uq_idx = topic_count(docs, "university")
        uq_title_variants = [
            title,                                      # short — let format_title pad if needed
            f"{title} at UQ Brisbane",
            f"The University of Queensland — {title}",
            f"{title} UQ Campus Services",
        ]
        raw_uq_title = uq_title_variants[uq_idx % len(uq_title_variants)]

        doc = {
            "doc_id": get_next_doc_id(docs),
            "title": format_title(raw_uq_title, "university"),
            "content": ensure_word_range(content, "university"),
            "source": "uq_official",
            "topic": "university",
            "url": url
        }

        docs.append(doc)

    print(f"University documents added: {topic_count(docs, 'university')}")


def validate_docs(docs):
    print("\nChecking generated corpus...")

    required_fields = ["doc_id", "title", "content", "source", "topic", "url"]
    allowed_sources = {"wikipedia", "tripadvisor", "tourism_au", "uq_official", "news", "local_curated"}
    allowed_topics = {"tourism", "restaurant", "university", "transport", "culture_food"}

    ids = [doc["doc_id"] for doc in docs]
    topic_counter = Counter(doc["topic"] for doc in docs)
    source_counter = Counter(doc["source"] for doc in docs)

    print(f"Total documents: {len(docs)}")
    print(f"Unique doc_ids: {len(set(ids))}")
    print(f"Topic distribution: {topic_counter}")
    print(f"Source distribution: {source_counter}")

    lengths = [len(doc["content"].split()) for doc in docs]

    if lengths:
        print(
            f"Content length: shortest {min(lengths)}, "
            f"longest {max(lengths)}, "
            f"average {sum(lengths) / len(lengths):.0f}"
        )

    for doc in docs:
        missing = [field for field in required_fields if field not in doc]
        if missing:
            print(f"❌ {doc.get('doc_id', '?')} missing fields: {missing}")

        if doc.get("source") not in allowed_sources:
            print(f"❌ {doc.get('doc_id', '?')} invalid source: {doc.get('source')}")

        if doc.get("topic") not in allowed_topics:
            print(f"❌ {doc.get('doc_id', '?')} invalid topic: {doc.get('topic')}")

        if doc.get("url") and not doc["url"].startswith("http"):
            print(f"⚠️ {doc.get('doc_id','?')} url may be invalid: {doc.get('url')}")

        word_count = len(doc["content"].split())
        if word_count < 100 or word_count > 300:
            print(f"⚠️ {doc['doc_id']} content length is {word_count} words")

        title_words = len(doc["title"].split())
        if title_words < 5 or title_words > 15:
            print(f"⚠️ {doc['doc_id']} title length is {title_words} words")


def save_docs(docs, path=None):
    out = path or OUTPUT_FILE
    with open(out, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(docs)} documents to {out}")


def load_existing_docs(path: str) -> list:
    """Load existing corpus if it exists, for incremental generation."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Generate brisbane_corpus.json for the Brisbane QA corpus."
    )
    parser.add_argument(
        "--topic-only",
        metavar="TOPIC",
        default=None,
        help=(
            "Only generate documents for a specific topic and merge into the "
            "existing corpus file. Example: --topic-only transport"
        ),
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help="Output file path (default: brisbane_corpus.json)",
    )
    args = parser.parse_args()

    topic_only = args.topic_only
    output_path = args.output

    VALID_TOPICS = {"tourism", "restaurant", "university", "transport", "culture_food"}

    if topic_only and topic_only not in VALID_TOPICS:
        print(f"❌ Unknown topic '{topic_only}'. Must be one of: {sorted(VALID_TOPICS)}")
        return

    if topic_only:
        # Incremental mode: load existing corpus, only generate the missing topic
        docs = load_existing_docs(output_path)
        existing_count = topic_count(docs, topic_only)
        target = TARGETS.get(topic_only, 0)
        print(f"Incremental mode: topic='{topic_only}'")
        print(f"  Existing: {existing_count}  /  Target: {target}")

        if existing_count >= target:
            print(f"✅ '{topic_only}' already meets target ({existing_count} >= {target}). Nothing to do.")
            return

        # Run only the relevant generator
        if topic_only == "restaurant":
            generate_restaurant_docs(docs)
        elif topic_only == "university":
            generate_uq_docs(docs)
        else:
            # Filter WIKIPEDIA_SOURCES to the requested topic only
            global WIKIPEDIA_SOURCES
            WIKIPEDIA_SOURCES = [s for s in WIKIPEDIA_SOURCES if s["topic"] == topic_only]
            generate_wikipedia_docs(docs)

        # Re-number doc_ids sequentially to avoid gaps/collisions
        for i, d in enumerate(docs, 1):
            d["doc_id"] = f"brisbane_{i:03d}"

    else:
        # Full generation mode
        docs = []
        generate_wikipedia_docs(docs)
        generate_restaurant_docs(docs)
        generate_uq_docs(docs)

    validate_docs(docs)
    save_docs(docs, path=output_path)


if __name__ == "__main__":
    main()