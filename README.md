# Team Project

## Description

The team project consists of two modules. Each module requires participants to apply the skills they have learned to date, and explore a dataset of their choosing. The first part of the team project involves creating a simple program with a database in order to analyze a dataset from an open source, such as Kaggle. In the second part of the team project, teams will come together again and apply the skills developed in each of the data science or machine learning foundations certificate streams. Teams will either create a data visualization or a machine learning model.

Participants will work in assigned teams of 4-5.

#### Project Descriptions

- [First Team Project Description](./team_project_1.md)
- [Second Team Project Description](./team_project_2.md)

## Learning Outcomes

By the end of Team Project Module 1, participants will be able to:

- Resolve merge conflicts
- Describe common problems or challenges a team encounters when working collaboratively using Git and GitHub
- Create a program to analyze a dataset with contributions from multiple team members

By the end of Team Project Module 2, participants will be able to:

- Create a data visualization as a team
- Create a machine learning model as a team

### Contacts

**Questions can be submitted to the _#cohort-3-help_ channel on Slack**

- Technical Facilitator:

  - **Phil Van-Lane**(he/him)
    phil.vanlane@mail.utoronto.ca

- Learning Support Staff:
  - **Taneea Agrawaal** (she/her)
    taneea@cs.toronto.edu
  - **Farzaneh Hashemi** (she/her )
    fhashemi.ma@gmail.com
  - **Tong Su** (she/her)
    tong.su@mail.utoronto.ca

### Delivery of Team Project Modules

Each Team Project module will include two live learning sessions and one case study presentation. During live learning sessions, facilitators will introduce the project, walk through relevant examples, and introduce various team skills that support project success. The remaining time will be used for teams to assemble and work on their projects, as well as get help from the facilitator or the learning support to troubleshoot any issues a team may be encountering.

Work periods will also be used as opportunities for teams to collaborate and work together, while accessing learning support.

### Schedule

| Day 1                 | Day 2                 | Day 3      | Day 4       | Day 5       |
| --------------------- | --------------------- | ---------- | ----------- | ----------- |
| Live Learning Session | Live Learning Session | Case Study | Work Period | Work Period |

## Requirements

- Participants are expected to attend live learning sessions and the case study as part of the learning experience. Participants are encouraged to use the scheduled work period time to complete their projects.
- Participants are encouraged to ask questions and collaborate with others to enhance learning.
- Participants must have a computer and an internet connection to participate in online activities.
- Participants must not use generative AI such as ChatGPT to generate code to complete assignments. It should be used as a supportive tool to seek out answers to questions you may have.
- We expect participants to have completed the [onboarding repo](https://github.com/UofT-DSI/onboarding/tree/main/onboarding_documents).
- We encourage participants to default to having their camera on at all times, and turning the camera off only as needed. This will greatly enhance the learning experience for all participants and provides real-time feedback for the instructional team.

### How to get help

![image](/steps-to-ask-for-help.png)

## Folder Structure

### Project 1

```markdown
|-- data
|---- processed
|---- raw
|---- sql
|-- reports
|-- src
|-- README.md
|-- .gitignore
```

### Project 2

```markdown
|-- data
|---- processed
|---- raw
|---- sql
|-- experiments
|-- models
|-- reports
|-- src
|-- README.md
|-- .gitignore
```

- **Data:** Contains the raw, processed and final data. For any data living in a database, make sure to export the tables out into the `sql` folder, so it can be used by anyone else.
- **Experiments:** A folder for experiments
- **Models:** A folder containing trained models or model predictions
- **Reports:** Generated HTML, PDF etc. of your report
- **src:** Project source code
- README: This file!
- .gitignore: Files to exclude from this folder, specified by the Technical Facilitator

<br>

# Team 13 Work

## Team members

- Angel
- Alison Wu
- Ernani Fantinatti
- Fredy Rincón
- James Li

## Roles

## Diagram

![Main Diagram](data/Team-13/Images/Diagram_01.jpg?raw=true "Diagram 01")

## Team 13 - Rules of Engagement

* Be open and transparent in your communication to ensure everyone shares information.
  * [X] Acknowledged - Angel
  * [x] Acknowledged - Alison Wu
  * [X] Acknowledged - Ernani Fantinatti
  * [X] Acknowledged - Fredy Rincón
  * [x] Acknowledged - James Li
* Clearly define each team member's role to avoid confusion and ensure everyone is accountable.
  * [X] Acknowledged - Angel
  * [x] Acknowledged - Alison Wu
  * [X] Acknowledged - Ernani Fantinatti
  * [X] Acknowledged - Fredy Rincón
  * [x] Acknowledged - James Li
* Encourage all team members to participate and respect different perspectives.
  * [X] Acknowledged - Angel
  * [x] Acknowledged - Alison Wu
  * [X] Acknowledged - Ernani Fantinatti
  * [X] Acknowledged - Fredy Rincón
  * [x] Acknowledged - James Li
* Address disagreements promptly and positively manage them.
  * [X] Acknowledged - Angel
  * [x] Acknowledged - Alison Wu
  * [X] Acknowledged - Ernani Fantinatti
  * [X] Acknowledged - Fredy Rincón
  * [x] Acknowledged - James Li
* Prioritize essential issues, stay focused, and make good use of time during meetings and collaborations.
  * [X] Acknowledged - Angel
  * [x] Acknowledged - Alison Wu
  * [X] Acknowledged - Ernani Fantinatti
  * [X] Acknowledged - Fredy Rincón
  * [x] Acknowledged - James Li

## Answering questions:

- What is the primary focus within the dataset?<br>
  - This dataset gives a detailed look at customer behavior on an e-commerce platform. Each record represents a unique customer, showing their interactions and transactions. The information helps analyze customer preferences, engagement, and satisfaction. Businesses can use this data to make informed decisions to improve the customer experience.<br>
- What are potential relationships in the data that you could explore?<br>
  - 1- `Items Purchased` against `Membership types`.<br>
  - 2- Analysis on Cities and Average income.<br>
  - 3- Purchase habits per Average Rating<br>
  - 4- <br>
- What are key questions your project could answer?<br>
  - Do customers with a Gold membership buy more items than those with Silver or Bronze memberships?<br>
  - How sensitive is each gender to customer satisfaction in relation to discounts being applied while purchasing?<br>
  - Which age group spends the most money on the platform?<br>
  - Are customers who receive discounts more satisfied than those who do not?<br>

## Questions to discuss when reviewing your dataset:

* What are the key variables and attributes in your dataset?<br>
  * sdadfsdfsd
  * Genger, Discount Applied and Satisfaction Level
* How can we explore the relationships between different variables?<br>

      * Using data visualizations such as boxplots and histograms to compare distributions, and statistical tests like ANOVA to identify significant differences. Regression models help quantify relationships between variables.
      * Using chi-square method
* Are there any patterns or trends in the data that we can identify?<br>
      * Customers with Gold and Silver memberships tend to purchase slightly more items than Bronze members.
      * Higher total spend is strongly associated with purchasing more items.
      * yes, for males, discounts seem to cause dissatisfaction, while for females, the response to discounts is mixed and might depend on other factors not captured in this dataset.
* Who is the intended audience for our data analysis?<br>
      * Marketing teams, business analysts, and decision-makers interested in understanding customer purchasing behavior and improving membership benefits.
      * Marketing department
* What is the question our analysis is trying to answer?<br>
      * sdadfsdfsd
      * The difference in sensitivity of genders reaction of discounts being applied or not
* Are there any specific libraries or frameworks that are well-suited to our project requirements?<br>
          * sdadfsdfsd
          * chi-square

**NOTE FOR CLASSMATES FROM FREDY:** Should we list all questions we all worked here?
- Do customers with a Gold membership buy more items than those with Silver or Bronze memberships?

- Are there any specific libraries or frameworks that are well-suited to our project requirements?
- 
- Pandas and NumPy for data manipulation, Matplotlib and Seaborn for visualization, and scikit-learn for statistical modeling and regression analysis are well-suited for our project.
<br> 
