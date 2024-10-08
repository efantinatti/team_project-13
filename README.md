# Team Project

## Description
The team project consists of two modules. Each module requires participants to apply the skills they have learned to date, and explore a dataset of their choosing. The first part of the team project involves creating a simple program with a database in order to analyze a dataset from an open source, such as Kaggle. In the second part of the team project, teams will come together again and apply the skills developed in each of the data science or machine learning foundations certificate streams. Teams will either create a data visualization or a machine learning model.

Participants will work in assigned teams of 4-5. 

#### Project Descriptions

* [First Team Project Description](./team_project_1.md)
* [Second Team Project Description](./team_project_2.md)

## Learning Outcomes
By the end of Team Project Module 1, participants will be able to:
* Resolve merge conflicts
* Describe common problems or challenges a team encounters when working collaboratively using Git and GitHub
* Create a program to analyze a dataset with contributions from multiple team members

By the end of Team Project Module 2, participants will be able to:
* Create a data visualization as a team
* Create a machine learning model as a team

### Contacts
**Questions can be submitted to the _#cohort-3-help_ channel on Slack**

* Technical Facilitator: 
  * **Phil Van-Lane**(he/him)
  phil.vanlane@mail.utoronto.ca

* Learning Support Staff:
  * **Taneea Agrawaal** (she/her)
  taneea@cs.toronto.edu
  * **Farzaneh Hashemi** (she/her )
  fhashemi.ma@gmail.com
  * **Tong Su** (she/her)
  tong.su@mail.utoronto.ca

### Delivery of Team Project Modules

Each Team Project module will include two live learning sessions and one case study presentation. During live learning sessions, facilitators will introduce the project, walk through relevant examples, and introduce various team skills that support project success. The remaining time will be used for teams to assemble and work on their projects, as well as get help from the facilitator or the learning support to troubleshoot any issues a team may be encountering. 

Work periods will also be used as opportunities for teams to collaborate and work together, while accessing learning support. 

### Schedule

|Day 1|Day 2|Day 3|Day 4|Day 5|
|-----|-----|-----|-----|-----|
|Live Learning Session |Live Learning Session|Case Study|Work Period|Work Period|

## Requirements
* Participants are expected to attend live learning sessions and the case study as part of the learning experience. Participants are encouraged to use the scheduled work period time to complete their projects.
* Participants are encouraged to ask questions and collaborate with others to enhance learning.
* Participants must have a computer and an internet connection to participate in online activities.
* Participants must not use generative AI such as ChatGPT to generate code to complete assignments. It should be used as a supportive tool to seek out answers to questions you may have.
* We expect participants to have completed the [onboarding repo](https://github.com/UofT-DSI/onboarding/tree/main/onboarding_documents).
* We encourage participants to default to having their camera on at all times, and turning the camera off only as needed. This will greatly enhance the learning experience for all participants and provides real-time feedback for the instructional team. 

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

* **Data:** Contains the raw, processed and final data. For any data living in a database, make sure to export the tables out into the `sql` folder, so it can be used by anyone else.
* **Experiments:** A folder for experiments
* **Models:** A folder containing trained models or model predictions
* **Reports:** Generated HTML, PDF etc. of your report
* **src:** Project source code
* README: This file!
* .gitignore: Files to exclude from this folder, specified by the Technical Facilitator

# Team 13, Project 1

## Team members
* Angel
* Alison Wu
* Ernani Fantinatti
* Fredy Rincón
* James Li

## Roles

## Diagram
![Main Diagram](data/Images/Diagram_01.jpg?raw=true "Diagram 01")

This diagram provides a detailed overview of an E-commerce Customer Behavior Dataset project workflow. Here’s a step-by-step explanation of each component:

1. Team Members:
  -	The project team consists of five members:
    - Angel Yang
    - Alison Wu
    - Ernani Fantinatti
    - Fredy Rincón
    - James Li

2. Data Source:
  - The dataset, "E-commerce Customer Behavior - Sheet1.csv," is sourced from Kaggle, a well-known platform for data science competitions and datasets.

3. Local Database:
  - The dataset is ingested into a local SQLite database. The main table created from the CSV has 11 columns and 350 rows.
  - The DB Browser tool is used to visualize and manage the SQLite database.

4. Database Schema:
  - The database contains several tables, such as:
     - E_Comm_Customer_Behavior

  - Manually added tables to improve our dataset:
     - Generations
     - ecommerce_sales
     - income_by_city
     - kaggle_income

5. Version Control and Collaboration:
  - GitHub is used for version control and team collaboration.
  - The team's interactions and code contributions are managed through individual branches for each team member (AYang, AlisonWu, EFantinatti, FredyRincon, JamesLi).
  - These individual branches are merged into the main project branch `team-project-1`.

6. Final Delivery:
   - The `team-project-1` branch is eventually merged into the `main` branch for the final delivery of the project 1

# ERD
![ERD](data/Images/Team13%20Project%201%20ERD.png?raw=true "ERD")

### Description:
Details of each component:

### Entities
1. **Customer**
   - **Attributes:**
     - CustomerID (Primary Key)
     - Name
     - Email
     - Phone
     - Address
   - **Description:**
     - Represents the customers using the system.

2. **Order**
   - **Attributes:**
     - OrderID (Primary Key)
     - OrderDate
     - TotalAmount
     - CustomerID (Foreign Key)
   - **Description:**
     - Represents the orders placed by customers. Each order is linked to a specific customer through CustomerID.

3. **Product**
   - **Attributes:**
     - ProductID (Primary Key)
     - Name
     - Description
     - Price
     - Stock
   - **Description:**
     - Represents the products available in the system.

4. **OrderDetail**
   - **Attributes:**
     - OrderDetailID (Primary Key)
     - OrderID (Foreign Key)
     - ProductID (Foreign Key)
     - Quantity
     - Price
   - **Description:**
     - Represents the details of each product within an order. Links to both Order and Product entities.

### Relationships
1. **Customer to Order**
   - **Type:** One-to-Many
   - **Description:** One customer can place multiple orders. This relationship is represented by CustomerID being a foreign key in the Order entity.

2. **Order to OrderDetail**
   - **Type:** One-to-Many
   - **Description:** One order can have multiple order details. This relationship is represented by OrderID being a foreign key in the OrderDetail entity.

3. **Product to OrderDetail**
   - **Type:** One-to-Many
   - **Description:** One product can appear in multiple order details. This relationship is represented by ProductID being a foreign key in the OrderDetail entity.

### Diagram Flow
1. **Customer Entity:**
   - Contains attributes related to customer information such as CustomerID, Name, Email, Phone, and Address.
   - Is related to the Order entity, indicating that customers can place orders.

2. **Order Entity:**
   - Contains attributes like OrderID, OrderDate, TotalAmount, and a foreign key CustomerID.
   - Is related to the OrderDetail entity, showing that an order consists of multiple order details.
   
3. **Product Entity:**
   - Contains attributes such as ProductID, Name, Description, Price, and Stock.
   - Is related to the OrderDetail entity, indicating that products can be part of multiple order details.

4. **OrderDetail Entity:**
   - Contains attributes like OrderDetailID, OrderID, ProductID, Quantity, and Price.
   - Links the Order and Product entities, showing which products are included in which orders.


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
* What is the primary focus within the dataset?<br>
    * This dataset gives a detailed look at customer behavior on an e-commerce platform. Each record represents a unique customer, showing their interactions and transactions. The information helps analyze customer preferences, engagement, and satisfaction. Businesses can use this data to make informed decisions to improve the customer experience.<br>
* What are potential relationships in the data that you could explore?<br>
    * 1- Analysis on Cities and Average income.<br>
    * 2- Purchase habits per Average Rating<br>
    * 3- `Membership Type` against `Items Purchased`. (See [Membership Level Purchase Analysis](./data/Code/Membership_Level_Purchase_Analysis.ipynb "Membership Level Purchase Analysis") for details)<br>
* What are key questions your project could answer?<br>
    * How sensitive is each gender to customer satisfaction in relation to discounts being applied while purchasing?<br>
    * Are age and gender variables statistically significant predictor of a high value customer?<br>
    * Do customers with a Gold membership buy more items than those with Silver or Bronze memberships?<br>
    * Are customers who receive discounts more satisfied than those who do not?<br>
    * Do customers with a Gold membership buy more items than those with Silver or Bronze memberships? (See [Membership Level Purchase Analysis](./data/Code/Membership_Level_Purchase_Analysis.ipynb "Membership Level Purchase Analysis") for details)<br>

## Questions to discuss when reviewing your dataset:

* What are the key variables and attributes in your dataset?<br>
  * Alison Wu:<br> Age, Gender and Transaction details
  * Angel Yang:<br> : Gender, Total Spend
  * Ernani Fantinatti: Age, Membership Type, Items Purchased, Average Rating, Generation, Satisfaction Level.<br>
  * Fredy Rincón: Membership Type, Items Purchased and Total Spend
  * James Li: Gender, Discount Applied and Satisfaction Level<br>
* How can we explore the relationships between different variables?<br>
  * Alison Wu:<br>: Relationships can be explored through correlation analysis or regression modeling
  * Angel Yang:<br>: we can use python to create correlation plot, or run codes utilizing pandas data package
  * Ernani Fantinatti: Yes, specially between Age, Membership Type and Satisfaction Level.<br>
  * Fredy Rincón: We can use visualizations like boxplots and histograms to compare distributions, and statistical tests like ANOVA to identify significant differences. Regression models help quantify relationships between variables.<br>
  * James Li: Using chi-square method<br>
* Are there any patterns or trends in the data that we can identify?<br>
  * Alison Wu:<br>: Patterns can include purchasing trends across different demographics
  * Angel Yang:<br>: overall, male customers spent more than female customer.
  * Ernani Fantinatti: Yes, Higher prices grows with age.<br>
  * Fredy Rincón: Customers with Gold and Silver memberships tend to purchase slightly more items than Bronze members.Also, higher total spend is strongly associated with purchasing more items.<br>
  * James Li: Yes, for males, discounts seem to cause dissatisfaction, while for females, the response to discounts is mixed and might depend on other factors not captured in this dataset.<br>
* Who is the intended audience for our data analysis?<br>
  * Alison Wu:<br>: The intended audience can include e-commerce businesses, marketing teams, and customer experience managers looking to optimize their strategies and improve customer satisfaction.
  * Angel Yang:<br>: marketing team of the ecommerce company 
  * Ernani Fantinatti: Companies interested in understanding the consumer market in general for Age group, generations and cities.<br>
  * Fredy Rincón: The intended audience includes marketing teams, business analysts, and decision-makers interested in understanding customer purchasing behavior and improving membership benefits.<br>
  * James Li: Marketing department<br>
* What is the question our analysis is trying to answer?<br>
  * Alison Wu:<br>: How different factors such as demographics, membership levels, and discounts influence customer behavior and satisfaction
  * Angel Yang:<br>: which demographic variables can be used as predictor of a high value customer 
  * Ernani Fantinatti: What age group are intending to spend more.<br>
  * Fredy Rincón: Do customers with a Gold membership buy more items than those with Silver or Bronze memberships?<br>
  * James Li: The difference in sensitivity of genders reaction of discounts being applied or not<br>
* Are there any specific libraries or frameworks that are well-suited to our project requirements?<br>
  * Alison Wu:<br>: Pandas, NumPy and Matplotlib
  * Angel Yang:<br> sklearn, pandas, matplotlib
  * Ernani Fantinatti: Pandas, matplotlib. SM Model Spec, one-hot-encoding, SQLite3.<br>
  * Fredy Rincón: Libraries like Pandas and NumPy for data manipulation, Matplotlib and Seaborn for visualization, and scikit-learn for statistical modeling and regression analysis are well-suited for our project.<br>
  * James Li: chi-square<br>

## Tables description

| Table | Description |
|-------|-------------|
| E_Comm_Customer_Behavior | Main table from the designated dataset. |
| income_by_city | Income by city, extracted from Kaggle. |
| Generations | Own created table with the generations by age bins. |

## Dataset Columns Description

| Column | Description |
| ------- | ---------- |
| Customer ID| A unique identifier for each customer. |
| Gender| The gender of the customer. |
| Age| The age of the customer. |
| City| The city where the customer lives. |
| Membership Type| The type of membership (Gold, Silver, Bronze). |
| Total Spend| The total amount spent by the customer. |
| Items Purchased| The number of items purchased by the customer. |
| Average Rating| The average rating given by the customer. |
| Discount Applied| Indicates whether a discount was applied. |
| Days Since Last Purchase| The number of days since the last purchase. |
| Satisfaction Level| The satisfaction level of the customer. |


## Videos:
  
  * [Alison Wu](https://drive.google.com/file/d/1SvlxO8eK2QBqxtPGW71PPRDMDFA2rZFg/view?usp=sharing "Alison Wu's video")
  * [Angel Yang](https://drive.google.com/file/d/1gMsrpWer4GmODwNPpfQICXMVi5FwGf0g/view?usp=drivesdk
  "Angel Yang's video")
  * [Ernani Fantinatti](https://youtu.be/8zqygKOwHVw "Ernani Fantinatti's video")
  * [Fredy Rincón](https://youtu.be/6n_e4El12xc?si=H-thyjlrLHvv2gSj "Fredy Rincón's video")
  * [James Li](https://drive.google.com/file/d/1nue-OUbawdU11nv_m7tIR7Wf-6MGzI15/view?usp=sharing "James Li's video")
 