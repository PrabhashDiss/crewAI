import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool

OPENAI_API_BASE="http://localhost:1234/v1"
OPENAI_MODEL_NAME="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
OPENAI_API_KEY="lm-studio"
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
os.environ["OPENAI_MODEL_NAME"] = OPENAI_MODEL_NAME
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

scrap_website_tool = ScrapeWebsiteTool()

researcher = Agent(
    role='Research Analyst',
    goal='Extract information from provided URL and synthesize it into an informative article',
    backstory="""You're a skilled research analyst adept at gathering information from various sources and distilling it into cohesive narratives.
    Your ability to synthesize complex information will be crucial in crafting an engaging article.""",
    verbose=True,
    allow_delegation=False,
    tools=[scrap_website_tool]
)

writer = Agent(
    role='Content Writer',
    goal='Craft a compelling article based on the extracted information',
    backstory="""You're an experienced content writer known for your ability to create engaging and informative articles.
    Your creativity and storytelling skills will be instrumental in producing an article that captivates readers.""",
    verbose=True,
    allow_delegation=False
)

# Create tasks for your agents
task_extract_information = Task(
    description="""Extract relevant information from the URL https://github.blog/2024-04-04-what-is-retrieval-augmented-generation-and-what-does-it-do-for-generative-ai/. 
    Summarize key points and gather data that will form the basis of the article.""",
    expected_output="Summary of extracted information and key data points",
    agent=researcher
)

task_write_article = Task(
    description="""Based on the extracted information, craft a well-written article in HTML format.
    Ensure that the article is informative, engaging, and formatted appropriately for online reading.""",
    expected_output="HTML document containing the article",
    agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task_extract_information, task_write_article],
    verbose=2,
    process=Process.sequential
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
