--ALTER TABLE "E-commerce Customer Behavior" RENAME TO E_Comm_Customer_Behavior;

--Initial evaluation
--SELECT * FROM E_Comm_Customer_Behavior;

-- List tables
SELECT name FROM sqlite_master WHERE type='table';

-- create an income by city table for the cities included in the ecommerce dataset
CREATE TABLE income_by_city 
as 
SELECT avg(median) as median_income, City
from kaggle_income
where City in (SELECT DISTINCT City
from ecommerce_sales)
group by City


--check whether a city's median income has any relationship to the type of membership its residents hold
SELECT count(*) as customer_count, "Membership Type", s.City, round(max(median_income),2) as median_income 
from ecommerce_sales s
left join income_by_city c on s.City = c.City
group by s.city, "Membership Type"
order by median_income DESC

