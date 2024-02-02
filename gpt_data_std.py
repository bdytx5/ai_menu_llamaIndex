import openai
import PyPDF2
import json
import re





def split_text(text, chunk_size=4000, overlap=500):
    chunks = []
    start = 0
    while start < len(text):
        if start + chunk_size > len(text):
            chunks.append(text[start:])
        else:
            end = start + chunk_size
            chunks.append(text[start:end + overlap])
        start += chunk_size
    return chunks

def read_and_process_menu(pdf_path):
    menu_data = []

    # Initialize the OpenAI client with your API key
    client = openai.OpenAI(api_key='sk-')

    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
        # for page_num in range(1):

            page_text = reader.pages[page_num].extract_text()

            # # Generating the prompt for OpenAI
            # response = client.chat.completions.create(
            #     model="gpt-3.5-turbo-1106",
            #     messages=[
            #         {"role": "system", "content": "You are a helpful assistant."},
            #         {"role": "user", "content": "I'm going to give you some text that I want you to convert into a JSON object (list), each item with a title and description. For example, if the text describes various dishes at a restaurant, the JSON output might look like: [{'title': 'Chicken Special', 'description': 'A delicious chicken dish seasoned with herbs and spices.'}, {'title': 'Seafood Platter', 'description': 'An assortment of fresh seafood, including shrimp, scallops, and lobster.'}] ONLY RESPOND WITH THE JSON OBJECT AND NOTHING ELSE. HERE IS THE DATA I WANT YOU TO CONVERT: {}".format(page_text)}
            #     ]
            # )
            text_chunks = split_text(page_text)

            for chunk in text_chunks:

                prompt_text = ("I'm going to give you some text that I want you to convert into a JSON object (list), "
                            "each item with a title and description. For example, if the text describes various dishes at a restaurant, "
                            "the JSON output might look like: [{'title': 'Chicken Special', 'description': 'A delicious chicken dish seasoned with herbs and spices.', 'keywords': 'chicken'}, "
                            "{'title': 'Seafood Platter', 'description': 'An assortment of fresh seafood, including shrimp, scallops, and lobster.', 'keywords': 'seafood, shrimp,platter'}]"
                            # "NOTE: THE ITEM TITLES WILL NOT BE SOMETHING GENERIC LIKE 'SANDWHICHES' -> THEY ARE ONLY SPECIC ITEMS, NOT CATEGORIES"
                            "NOTE: THE ITEM TITLES WILL NOT BE A CATEGORY OF ITEMS, RATHER THEY ARE SPECIFIC DISHES/ENTRES/APETIZERS/DRINKS ETC. -> THEY ARE ONLY SPECIC ITEMS, NOT CATEGORIES"
                            "NOTE: Keywords should be the a 2-4 categories/common search terms describing the item, for example (but not limited to) -> dessert, side, chicken, sandwhich, cocktail, drink, burger, salad, etc etc -> Try to use multiple categories/descriptors"
                            "THIS DATA IS CHUNKED SO IF THE DATA AT THE BEGINNING OR END (FOR TITLE OR DESCRIPTION) SEEMS INCOMPLETE COMPARED TO THE REST OF THE ENTRIES, JUST SKIP IT"
                            "ONLY RESPOND WITH THE JSON OBJECT AND NOTHING ELSE. CONVERT THE FULL DATA. DO NOT TRUNCATE OR STOP EARLY. FULL TEXT! HERE IS THE DATA I WANT YOU TO CONVERT: " + chunk)

                response = client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_text}
                    ]
                )

                # Extract the response and format it as required
                if response.choices:
                    assistant_reply = response.choices[0].message.content
                    try:
                        # Attempt to parse the reply as JSON
                        print(assistant_reply)
                        match = re.search(r"```json(.+?)```", assistant_reply, re.DOTALL)
                        if match:
                            cleaned_reply = match.group(1).strip()  # Extract the text between the delimiters
                        else:
                            cleaned_reply = ""  # No match found, set cleaned_reply to an empty string

                        json_reply = json.loads(cleaned_reply)


                        if isinstance(json_reply, list):
                            for item in json_reply:
                                item['page'] = page_num + 1
                            menu_data.extend(json_reply)
                        else:
                            print("Warning: JSON reply is not a list. It has been skipped.")

                    except json.JSONDecodeError as e:

                        # If parsing fails, use the whole response as description
                        print("JSON parsing failed with error:", e)

                        print("parsing failed. Try adjusting the prompt or creating this JSON Manually")
                else:
                    menu_data.append({
                        "page": page_num + 1,
                        "title": "Page " + str(page_num + 1),
                        "description": "No response"
                    })
    unique_menu_data = {}
    for item in menu_data:
        if item['title'] not in unique_menu_data:
            unique_menu_data[item['title']] = item

    # Convert the dictionary back to a list
    menu_data = list(unique_menu_data.values())

    return menu_data
# Replace 'path_to_your_pdf.pdf' with the actual path to your PDF file
pdf_path = './menu.pdf'
menu_items = read_and_process_menu(pdf_path)

# Saving the results to a JSON file
with open('gpt4_menu_data.json', 'w') as json_file:
    json.dump(menu_items, json_file, indent=4)
