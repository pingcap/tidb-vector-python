# DSPy Demo

This example demonstrates how to use the DSPy and TiDB Serverless to build a simple RAG application.

## Prerequisites

- A running TiDB Serverless cluster
- Python 3.10 or later
- Ollama or OpenAI

## Run the example

### Clone this repo

```bash
git clone https://github.com/pingcap/tidb-vector-python.git
```

### Create a virtual environment

```bash
cd tidb-vector-python/examples/dspy-demo
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set the environment variables

Get the TiDB connection string via `TIDB_HOST`, `TIDB_USERNAME`, and `TIDB_PASSWORD` from the TiDB Cloud console, as
described in the [Prerequisites](../README.md#prerequisites) section.

The TiDB connection string will look like:

```
mysql+pymysql://{TIDB_USER}:{TIDB_PASSWORD}@{TIDB_HOST}:{TIDB_PORT}/{TIDB_DB_NAME}?ssl_verify_cert=True&ssl_verify_identity=True
```

### Run this example

```text
$ python3 example.py
Connected to TiDB.
describe table: {'success': True, 'result': 6, 'error': None}
Initializing the TidbRM model...
TidbRM model initialized successfully.
Loading sample data...
sample_data.txt found.
Sample data loaded successfully.
Embedding sample data...
0 At My Wind [-0.27386308 -0.3816067  -0.12257734  0.04750763 -0.12517984]
1 Little Win [ 0.05535038 -0.2605278  -0.19080743 -0.3411712  -0.0255685 ]
2 Storm Wind [-0.24868685 -0.21516131 -0.03831396  0.08118728 -0.05171517]
3 Yes I Am ( [ 0.07458089 -0.31562874 -0.14104412 -0.13799803 -0.02719649]
4 The Great  [-0.08352712  0.12166582 -0.07781561  0.2473993  -0.3156342 ]
5 Rosario Da [-0.02921938  0.16053236 -0.2157185  -0.14237025  0.3970173 ]
6 Robert B.  [-0.05834749 -0.16446972 -0.00786973  0.02972636  0.03525066]
7 Richard M. [-0.13012317 -0.20805678  0.0064573   0.05393503  0.043081  ]
8 Everything [ 0.09028038  0.03007011 -0.29266015 -0.27439988 -0.2159805 ]
9 Everything [-0.02172723  0.22668567 -0.17105839  0.04179271 -0.18812893]
10 Janick Ger [ 0.179568   -0.4577289  -0.05370283  0.09678644 -0.27309376]
11 Dave Murra [ 0.01501587 -0.32756883 -0.08704209 -0.07916276 -0.23352458]
12 Roy Z | Ro [ 0.12954581 -0.27150235 -0.0992474   0.14631633  0.09378276]
13 Heather Ba [ 0.07651925 -0.23409796 -0.03234328  0.01846722 -0.09262329]
14 Gianfranco [ 0.00602041  0.29790103 -0.2082347   0.12557846  0.13808164]
Sample data embedded successfully.
Sample data number: 15
Inserting documents into TiDB...
Documents inserted successfully.
Answering the question: 'who write At My Window'...
Townes Van Zandt wrote At My Window, which is an album released in 1987. The reasoning provided explains that Townes Van Zandt was the songwriter behind this album, showcasing his unique style and poetic lyrics. The album features a mix of folk, country, and Americana sounds, exploring themes of love, loss, and self-discovery. At My Window is often cited as one of the greatest albums in American music history, having had a significant influence on many other artists.



Answer questions with short factoid answers.

---

Follow the following format.

Context: may contain relevant facts

Question: ${question}

Reasoning: Let's think step by step in order to ${produce the answer}. We ...

Answer: often between 1 and 5 words

---

Context:
[1] «{'long_text': 'Rosario Dawson | Rosario Isabel Dawson (born May 9, 1979) is an American actress, producer, singer, comic book writer, and political activist. She made her film debut in the 1995 teen drama "Kids". Her subsequent film roles include "He Got Game", "Men in Black II", "25th Hour", "Rent", "Sin City", "Death Proof", "Seven Pounds", "", and "Top Five". Dawson has also provided voice-over work for Disney and DC.'}»
[2] «{'long_text': 'Dave Murray (musician) | David Michael "Dave" Murray (born 23 December 1956) is an English guitarist and songwriter best known as one of the earliest members of the British heavy metal band Iron Maiden. Along with the group\'s bassist and primary songwriter Steve Harris, Murray has appeared on all of the band\'s releases.'}»
[3] «{'long_text': 'Heather Baker | Heather Baker (born October 9, 1984) is a female American songwriter, guitarist, producer and founder of the Electronica band Antiwave. Heather is known for being a session and touring guitar player for the likes of Bonnie Mckee (Pulse Music Publishing), Krewella (Columbia Records), Kerli (Island Records), The Iron Maidens (Powerslave Records) and currently plays with the band Fake Figures (members of Atreyu, Eyelid and Scars of Tomorrow) and NoMBe (TH3RD BRAIN)'}»
[4] «{'long_text': 'Janick Gers | Janick Robert Gers ( ; born 27 January 1957 in Hartlepool, England) is an English musician, best known for being one of the three current guitarists in Iron Maiden, along with Dave Murray and Adrian Smith, as well as his earlier work with Gillan and White Spirit.'}»
[5] «{'long_text': 'Robert B. Sherman | Robert Bernard Sherman (December 19, 1925 – March 6, 2012) was an American songwriter who specialized in musical films with his brother Richard Morton Sherman. According to the official Walt Disney Company website and independent fact checkers, "the Sherman Brothers were responsible for more motion picture musical song scores than any other songwriting team in film history." Some of the Sherman Brothers\' best known songs were incorporated into live action and animation musical films including: "Mary Poppins", "The Jungle Book", "The Many Adventures of Winnie the Pooh", "Chitty Chitty Bang Bang", "The Slipper and the Rose", and "Charlotte\'s Web". Their most well known work, however, remains the theme park song "It\'s a Small World (After All)". According to Time.com, this song is the most performed song of all time.'}»
[6] «{'long_text': 'Richard M. Sherman | Richard Morton Sherman (born June 12, 1928) is an American songwriter who specialized in musical films with his brother Robert Bernard Sherman. According to the official Walt Disney Company website and independent fact checkers, "the Sherman Brothers were responsible for more motion picture musical song scores than any other songwriting team in film history." Some of the Sherman Brothers\' best known songs were incorporated into live action and animation musical films including: "Mary Poppins", "The Jungle Book", "The Many Adventures of Winnie the Pooh", "Chitty Chitty Bang Bang", "Snoopy Come Home", "Bedknobs and Broomsticks", "The Slipper and the Rose", and "Charlotte\'s Web". Their most well known work, however, remains the theme park song "It\'s a Small World (After All)". According to Time.com, this song is the most performed song of all time.'}»
[7] «{'long_text': 'Everything Changes (Julian Lennon album) | Everything Changes is the sixth studio album by English singer-songwriter Julian Lennon. It was released on 2 October 2011.'}»
[8] «{'long_text': 'Roy Z | Roy Z (born February, 1968) is an American guitarist, songwriter and producer, best known for his work with Bruce Dickinson (from Iron Maiden), Halford, and Judas Priest. He also is the founder of Tribe of Gypsies, a Latin influenced hard rock band.'}»
[9] «{'long_text': 'Gianfranco Rosi (director) | Gianfranco Rosi is an Italian director, cinematographer, producer and screenwriter. His film "Sacro GRA" won Golden Lion at 70th Venice International Film Festival. "Sacro GRA" is the first documentary film to win Golden Lion in history of the Venice film festival and the first Italian film to win in fifteen years, after Gianni Amelio\'s "The Way We Laughed" won the award in 1998. His 2016 film "Fire at Sea", a documentary focused on European migrant crisis on the Sicilan island of Lampedusa, won the Golden Bear at the 66th Berlin International Film Festival. Rosi is the only documentary filmmaker to win two top prizes at major European film festivals (Cannes, Berlin and Venice) and is currently the only filmmaker besides Michael Haneke, Jafar Panahi, Ang Lee, and Ken Loach to win two top European festival prizes in the 21st century.'}»
[10] «{'long_text': 'The Great Victorian Collection | The Great Victorian Collection, published in 1975, is a novel by Northern Irish-Canadian writer Brian Moore. Set in Carmel, California, it tells the story of a man who dreams that the empty parking lot he can see from his hotel window has been transformed by the arrival of a collection of priceless Victoriana on display in a vast open-air market. When he awakes he finds that he can no longer distinguish the dream from reality.'}»
[11] «{'long_text': 'Everything Has Changed | "Everything Has Changed" is a song written and performed by American singer-songwriter Taylor Swift and English singer-songwriter Ed Sheeran, taken from Swift\'s fourth studio album, "Red" (2012). Produced by Butch Walker, the track was released as the sixth single from the album on July 16, 2013. "Everything Has Changed" is a guitar ballad combining folk and pop genres about "wanting to get to know a new lover better".'}»
[12] «{'long_text': 'Storm Windows | Storm Windows is the seventh album by American folk singer and songwriter John Prine, released in 1980. It was his last release on a major label – he would next join Al Bunetta and Dan Einstein to form Oh Boy Records on which all his subsequent recordings were released.'}»
[13] «{'long_text': 'Little Window | Little Window is the debut album of American singer-songwriter Baby Dee. The album was released in 2002 on the Durtro label. It was produced, composed, and performed entirely by Dee.'}»
[14] «{'long_text': 'Yes I Am (Melissa Etheridge album) | Yes I Am is the fourth studio album by American singer-songwriter Melissa Etheridge, released by Island Records on September 21, 1993 (see 1993 in music). The title is generally thought to refer to Etheridge\'s recent coming out as a lesbian, confirming long-standing rumors about her personal life. This is the album that gave Etheridge national recognition. The rock ballad "Come to My Window" was the first single released from the album, which peaked at No. 25 on the "Billboard" Hot 100, and its video featured actress Juliette Lewis having a nervous breakdown. This single brought the album into the public consciousness and was quickly followed by "I\'m the Only One", which became a major hit and reached No. 8 on the Hot 100, and "If I Wanted To", which hit No. 16.'}»
[15] «{'long_text': 'At My Window (album) | At My Window is an album released by Folk/country singer-songwriter Townes Van Zandt in 1987. This was Van Zandt\'s first studio album in the nine years that followed 1978\'s "Flyin\' Shoes", and his only studio album recorded in the 1980s. Although the songwriter had become less prolific, this release showed that the quality of his material remained high.'}»

Question: who write At My Window

Reasoning: Let's think step by step in order to At My Window is an album written and performed by Townes Van Zandt. The album was released in 1987 and it's considered one of his best works, showcasing his unique songwriting style and poetic lyrics. The album features a mix of folk, country, and Americana sounds, with songs that explore themes of love, loss, and self-discovery. At My Window is often cited as one of the greatest albums in the history of American music, and it has had a significant influence on many other artists. Townes Van Zandt was an American singer-songwriter who was active from the 1960s until his death in 1997. He was known for his poetic lyrics and his ability to tell stories through his

Answer: Townes Van Zandt wrote At My Window, which is an album released in 1987. The reasoning provided explains that Townes Van Zandt was the songwriter behind this album, showcasing his unique style and poetic lyrics. The album features a mix of folk, country, and Americana sounds, exploring themes of love, loss, and self-discovery. At My Window is often cited as one of the greatest albums in American music history, having had a significant influence on many other artists.
```
 