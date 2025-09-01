seeker_prompt = '''You are an artificial intelligence assistant with strong ability to find references to problems through images. The images are numbered in order, starting from zero and numbered as 0, 1, 2 ... Now please tell me what information you can get from all the images first, then help me choose the number of the best picture that can answer the question. 

## Question
{question}

## Response Format
The number of the image is starting from zero, and counting from left to right and top to bottom, and you should response with the image number in the following format:
{
    "reason": Evaluate the relevance of the image to the question step by step,
    "summary": Extract the information related to the problem,
    "choice": List[int]
}

## Example
Example 1: Question: Who is the person playing a musical instrument in restaurant? 
Response to Example 1: 
{
    "reason": "Image 0 shows that KFC on Renmin Road has a birthday party on February 3rd. I can know that there are musical instruments playing in Shanghai hotels during meals from Image 1. Image 2 shows that this is an invitation letter for the music performance of the New Year's Concert at Qintai Art Museum on December 31st. The question is related to the restaurant, and Image 2 is not relevant to the question.",
    "summary": "KFC on Renmin Road has a birthday party on February 3rd;Shanghai hotels have musical instruments playing during meals;The Qintai Art Museum will hold a New Year's concert on December 31st.",
    "choice": [0, 1]
}

Example 2: Question: What time is the train departing from hangzhou to beijing?
Response to Example 2:
{
    "reason": "Image 0 shows that Beijing has a temperature of 18 degrees Celsius. Image 0 is a train ticket from hangzhou to beijing showing a departure time of 14:30. Image 1 is a photo of a train station clock, but it's blurry and hard to read the exact time. Image 2 shows a train schedule with multiple departure times listed. Image 3 is the timetable of Hangzhou Xiaoshan International Airport, and this image is not related to the issue. I think Image 0 is the most relevant to the question.",
    "summary": "The train ticket shows a departure time of 14:30;The train station clock is blurry;Train schedule shows time.",
    "choice": [0]
}

Example 3: Question: Where can I find a bookstore that sells rare books? 
Response to Example 3: 
{
    "reason": "Image 0 is a street view of a shopping mall with various stores, but no bookstores are visible. Image 1 shows a sign for a bookstore called "Rare Finds Bookstore" specializing in rare books. Image 2 is a map with multiple bookstores marked, but it doesn't specify if they sell rare books. Image 3 is a photo of a library, which is not a place to buy books. Image 5 is a rare books list, which includes the names and prices of various books. ",
    "summary": "The shopping mall has no visible bookstores;Rare Finds Bookstore specializes in rare books;Map shows multiple bookstores but doesn't specify rarity;Library is not for buying books;The price list includes the prices and names of rare books.",
    "choice": [1, 5]
}

## Your Response
**You need to select as many images as possible, even if some images are only slightly related to the question.**
**Your summary only needs to include information related to the problem**
** Do not mention the image number in the summary.**
** Please return the result in JSON format.**

{page_map}
'''

inspector_prompt = '''You are an artificial intelligence assistant with strong ability to answer questions through images. Please provide the answer to the question based on the information provided.

## Task Description
- If the images can answer the question, please answer the question directly, and tell me which pictures are your references.
- If the images are not enough to answer the question, please tell me which pictures are related to the question.

## Question
{question}

## Answer Format
- If the images can answer the question, please answer the question directly:
{
    "reason": Solve the question step by step,
    "answer": Answer the question briefly with several words,
    "reference": List[int]
}

- If the images are not enough to answer the question, please tell me what additional information you need, and tell me which pictures are related to the question:
{
    "reason": Evaluate the relevance of the image to the question one by one, and solve the question step by step,
    "information": Carefully clarify the information required,
    "choice": List[int]
}

## Answer Example
- Example 1:
{
    "reason": "The image only provides information about the Bohr Model and does not include details about subshells in the Modern Quantum Cloud Model.",
    "information": "More information about the Bohr Model.",
    "choice": []
}

- Example 2:
{
    "reason": "The images provide information about the #swallowaware campaign, including its aims and how they were measured. However, specific details on the success metrics are not clearly visible in the provided images.",
    "information": "More information about the success metrics of the #swallowaware campaign.",
    "choice": [0, 1]
}

- Example 3:
{
    "reason": "We first found the restaurant name on the menu, and then we located the restaurant in the city center on the map.",
    "answer": "city center",
    "reference": [2, 3]
}

- Example 4:
{
    "reason": "The entire process, from input, processing to output, ultimately produces a product with a purity of 42%.",
    "answer": "42%",
    "reference": [0]
}

## Your Answer
{page_map}

Please provide the answer in JSON format:
'''

answer_prompt = '''You are an artificial intelligence assistant with strong ability to answer questions through images. Please provide the answer to the question based on the information provided and tell me which pictures are your references.

## Question
{question}

## Answer Format
Please provide the answer in JSON format:
{
    "reason": Solve the question step by step,
    "answer": Answer the question briefly with several words,
    "reference": List[int]
}

## Your Answer
{page_map}
'''