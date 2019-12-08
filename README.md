# Spark project MS Big Data Télécom : Kickstarter campaigns

Spark project for MS Big Data Telecom based on Kickstarter campaigns 2019-2020

Indication de lancement du projet : 

- Se placer dans le même dossier que le fichier build.sbt 
- Executer la commande : sbt assembly 
- Lancer son cluster spark 
- Executer la commande en remplaçant $PATH_TO_SPARK par le chemin vers votre spark et "Classe à lancer" par Preprocessor ou Trainer : 
> $PATH_TO_SPARK/bin/./spark-submit --class paristech."Classe à lancer" target/scala-2.11/spark_project_kickstarter_2019_2020-assembly-1.0.jar

