from crewai import Agent, Task, Crew
import os

os.environ["OPENAI_API_KEY"] = "your-new-key"

coder = Agent(
    role="Python Developer",
    goal="Write clean and efficient Python code",
    backstory="A genius coder who fixes bugs instantly"
)

task = Task(
    description="Write a Python function to check if a number is prime",
    agent=coder
)

crew = Crew(
    agents=[coder],
    tasks=[task]
)

result = crew.kickoff()
print(result)