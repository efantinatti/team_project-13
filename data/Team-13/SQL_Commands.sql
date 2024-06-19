--ALTER TABLE "E-commerce Customer Behavior" RENAME TO E_Comm_Customer_Behavior;

--Initial evaluation
--SELECT * FROM E_Comm_Customer_Behavior;

-- List tables
SELECT name FROM sqlite_master WHERE type='table';