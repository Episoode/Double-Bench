GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "en"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PROMPTS["DEFAULT_ENTITY_TYPES"] = ["Person", "Organization", "Place", "Time", "Event", "Concept", "Object", "Data", "Attribute", "Behavior"]

PROMPTS["entity_extraction"] = """
- Target Activity -
You are an AI assistant specializing in identifying **named entities** and relationships between **named entities**. Strictly follow the requirements during processing.
**Your output language of entities and relationships must be the same as the language of  input text**

- Goal -
Given a text document potentially related to this activity and a list of entity types, identify all **named entities** of these types from the text and all relationships between the identified **named entities**. These **named entities** must be specific, meaningful concepts, not just common words.
Do not provide information without supporting evidence.
Strictly adhere to all formatting requirements. Allow yourself to rethink before outputting.
**Your output language of entities and relationships must be the same as the language of  input text**

- Steps -
1.  First, Carefully interpret the provided text, accounting for OCR errors, noise, and inconsistencies. Disregard scanning artifacts and non-meaningful characters or sequences.
2.  Based on your understanding, identify significant and meaningful **named entities** of the specified types, focusing on specific concepts/categories (for examples，concepts, individuals, organizations, locations, dates，data，but not arbitrary words). When identifying from OCR text, prioritize likely intended names over misspellings.
    For each identified **named entity**, extract the required information with the following keys. Do not extract if the entity is unclear, noise, or lacks specific meaning after interpretation.
    node_name: The name of the **named entity**, using the most probable correct form or the form as it appears if interpretable, using the same language as the input text.
    node_type: Must be "entity".
    name_type: Belongs to one of the following entity types: [{entity_types}]
    attributes: Comprehensive description of the entity's attributes and activities
    **Set the format for each entity as (<node_type>{tuple_delimiter}<node_name>{tuple_delimiter}<name_type>{tuple_delimiter}<attributes>)**

3.  From the meaningful entities identified in Step 2, identify significant (source_entity, target_entity) pairs to each other within the interpreted context.
    For each pair of entities considered to be clearly related, extract the following information:
    - source_entity: The source entity name identified in Step 2.
    - target_entity: The target entity name identified in Step 2.
    - relationship_description: Explain the nature of the relationship between the source entity and the target entity concisely and precisely. Formulate the explanation as a clear subject-verb-object statement that describes the action, state, or connection. Crucially, the description *must* embed enough context about both the source and target entities (e.g., their entity types or defining characteristics) so that the relationship is completely understandable and actionable even without referring back to the original text context. Disregard literal co-occurrence artifacts; focus on the meaningful conceptual link.
    - relationship_keywords: One or more high-level keywords that summarize the overall nature of the relationship, focusing on concepts or themes rather than specific details. These keywords are for summarizing, not replacing the specific description.
    - relationship_strength: A numerical score representing the strength of the relationship between the source entity and the target entity.
    **Set the format for each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>).**

4.  Identify high-level keywords that summarize the main concepts, themes, or topics of the entire document based on your interpretation. These should capture the overall ideas in the document.
    Set the format for content-level keywords as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

5.  Return the output as a single list of all entities and relationships identified in Steps 2 and 3. Use **{record_delimiter}** as the list delimiter.

6.  **It is strictly forbidden to extract existing words as named entities** Ensure all identified named entities and relationships are significant, meaningful, and relevant within the document's interpreted context, specifically filtering out artifacts, noise, or inconsequential elements originating from the OCR process.

7.  When finished, output {completion_delimiter}.**Your output language of entities and relationships must be the same as the language of  input text**

######################
---Example---
######################
Entity_types: ["Product", "Feature", "Company", "Event", "Technology", "Specification"]
Text:  
The Vision Pro, developed by Apple, is a revolutionary augmented reality headset launched in 2023. 
The product integrates cutting-edge technologies such as spatial computing, eye-tracking, and hand gesture recognition. 
It offers users an immersive experience with a 4K micro-OLED display for each eye. Vision Pro supports seamless integration with existing Apple devices and provides advanced applications for gaming, productivity, and virtual meetings. 
The product launch event at WWDC 2023 marked a significant milestone for Apple, capturing global attention. The retail price starts at $3,499, and it is expected to set a new standard in the AR industry.

######################
Output:  
("entity"{tuple_delimiter}"Vision Pro"{tuple_delimiter}"Product"{tuple_delimiter}"Vision Pro is a revolutionary augmented reality headset developed and launched,integrating cutting-edge technologies and offering an immersive experience with a 4K micro-OLED display."{record_delimiter}
("entity"{tuple_delimiter}"Apple"{tuple_delimiter}"Company"{tuple_delimiter}"Apple is the company that developed Vision Pro and held WWDC 2023."{record_delimiter}
("entity"{tuple_delimiter}"WWDC 2023"{tuple_delimiter}"Event"{tuple_delimiter}"WWDC 2023 is an event held by Apple where the Vision Pro was launched in 2023."{record_delimiter}
("entity"{tuple_delimiter}"spatial computing"{tuple_delimiter}"Technology"{tuple_delimiter}"Spatial computing is a cutting-edge technology integrated into Vision Pro."{record_delimiter}
("entity"{tuple_delimiter}"4K micro-OLED display"{tuple_delimiter}"Specification"{tuple_delimiter}"The 4K micro-OLED display is a specification offered by Vision Pro for each eye."{record_delimiter}
("entity"{tuple_delimiter}"retail price"{tuple_delimiter}"Specification"{tuple_delimiter}"The retail price is a specification for Vision Pro that starts at $3,499."{record_delimiter}
("entity"{tuple_delimiter}"$3,499"{tuple_delimiter}"Specification"{tuple_delimiter}"$3,499 is the starting value for the retail price of Vision Pro."{record_delimiter}
("relationship"{tuple_delimiter}"Apple"{tuple_delimiter}"Vision Pro"{tuple_delimiter}"The company Apple developed the revolutionary augmented reality headset Vision Pro."{tuple_delimiter}"Development, Manufacturer"{tuple_delimiter}"9"){record_delimiter}
("relationship"{tuple_delimiter}"Vision Pro"{tuple_delimiter}"WWDC 2023"{tuple_delimiter}"The Vision Pro headset was launched during the WWDC 2023 event."{tuple_delimiter}"Launch Event, Timing"{tuple_delimiter}"10"){record_delimiter}
("relationship"{tuple_delimiter}"Vision Pro"{tuple_delimiter}"spatial computing"{tuple_delimiter}"The Vision Pro headset integrates cutting-edge spatial computing technology."{tuple_delimiter}"Integration, Technology"{tuple_delimiter}"8"){record_delimiter}
("relationship"{tuple_delimiter}"Vision Pro"{tuple_delimiter}"4K micro-OLED display"{tuple_delimiter}"The Vision Pro headset offers users an immersive experience with a 4K micro-OLED display."{tuple_delimiter}"Component, Feature"{tuple_delimiter}"9"){record_delimiter}
("relationship"{tuple_delimiter}"retail price"{tuple_delimiter}"$3,499"{tuple_delimiter}"The retail price specification for Vision Pro starts at the value $3,499."{tuple_delimiter}"Value, Starting Price"{tuple_delimiter}"10"){record_delimiter}
("relationship"{tuple_delimiter}"Apple"{tuple_delimiter}"WWDC 2023"{tuple_delimiter}"The company Apple held the event WWDC 2023."{tuple_delimiter}"Event Host, Association"{tuple_delimiter}"9"){record_delimiter}
("content_keywords"{tuple_delimiter}"Vision Pro, Apple, Augmented Reality, WWDC 2023, Launch, Technologies, Display, Pricing"){record_delimiter}{completion_delimiter}
#############################
- Task Data -  
(**Your output language of entities and relationships must be the same as the language of  input text**)
######################  
Entity_types: {entity_types}  
Text: {input_text}  
######################  
Output:  
"""

PROMPTS["entity_flit"] = """
You are tasked with filtering and refining a set of entities and relationships based on specificity, clarity, and relevance. Follow these specific guidelines:

For named-entities:
1. Keep named-entities that represent specific, specialized concepts, proper nouns, technical terms, or unique identifiers relevant to the text's domain.
2. Filter out common words, generic phrases, or entities that lack specific meaning or context within the domain.
3. Named-entities must still be clearly referred to and represented after being separated from the current context, and has a clear and specific meaning.

For relationships:
1. Filter out relationships that do not clearly connect exactly two distinct entities.
2. Keep relationships that describe a specific, unambiguous, and meaningful connection, action, attribute, or role between the two entities.
3. Remove vague, generic, or non-descriptive relationships.
4. If an entity is filtered out based on the above entity rules, ALL relationships involving that entity must also be removed.

Input and output format:
- Maintain the exact same format as the input,**Do not output other explanations and content**
- Your output should contain fewer entities and relationships based on the filtering criteria
- Do not change the structure or delimiters of the records

- Task Data -
{input_text}

Output:
"""

PROMPTS["summarize_entity_descriptions"] = """You are a helpful assistant responsible for generating a comprehensive summary for the data provided below.  
Given one or two entities and a list of descriptions, all entities are related to the same entity or group of entities. Please consolidate all this information into a comprehensive description.
If the provided descriptions contradict each other, resolve these contradictions and provide a single, coherent summary. Ensure it is written in third person and includes the entity names so that we have the full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

