-- MySQL dump 10.13  Distrib 8.0.31, for Win64 (x86_64)
--
-- Host: localhost    Database: wbanalysissystems
-- ------------------------------------------------------
-- Server version	5.7.40

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `admin`
--

DROP TABLE IF EXISTS `admin`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `admin` (
  `Id` int(11) NOT NULL AUTO_INCREMENT,
  `UserName` varchar(50) CHARACTER SET utf8 DEFAULT NULL,
  `PWD` varchar(50) CHARACTER SET utf8 DEFAULT NULL,
  PRIMARY KEY (`Id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `admin`
--

LOCK TABLES `admin` WRITE;
/*!40000 ALTER TABLE `admin` DISABLE KEYS */;
INSERT INTO `admin` VALUES (1,'admin','123456');
/*!40000 ALTER TABLE `admin` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `hotseacher`
--

DROP TABLE IF EXISTS `hotseacher`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `hotseacher` (
  `Id` bigint(100) NOT NULL AUTO_INCREMENT,
  `Titile` varchar(45) CHARACTER SET utf8 DEFAULT NULL,
  `Heat` varchar(45) CHARACTER SET utf8 DEFAULT NULL,
  `HotTimes` varchar(45) CHARACTER SET utf8 DEFAULT NULL,
  PRIMARY KEY (`Id`)
) ENGINE=InnoDB AUTO_INCREMENT=24 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `hotseacher`
--

LOCK TABLES `hotseacher` WRITE;
/*!40000 ALTER TABLE `hotseacher` DISABLE KEYS */;
INSERT INTO `hotseacher` VALUES (1,'90%学生被取消学籍系因游戏','329.4','2022-09-04 07:12:32'),(2,'成都疫情防控','122.6','2022-09-04 07:12:32'),(3,'2022服贸会观展指北','121.6','2022-09-04 07:12:32'),(4,'女子在永辉超市买2斤猪肉花103元','121.2','2022-09-04 07:12:32'),(5,'陈若仪回应儿子车祸后第一句话','105.4','2022-09-04 07:12:32'),(6,'RNG与LNG决胜局','92.8','2022-09-04 07:12:32'),(7,'不会又让二追三吧','86.5','2022-09-04 07:12:32'),(8,'世界游泳冠军遭酒店救生员暴力威胁','82.8','2022-09-04 07:12:32'),(9,'赵丽颖直播','66.1','2022-09-04 07:12:32'),(10,'河南变身全球最大人造钻石生产地','49.1','2022-09-04 07:12:32'),(11,'女子和男友吵架吃60粒褪黑素进ICU','45.9','2022-09-04 07:12:32'),(12,'恒力石化董事长成中国女首富','43.4','2022-09-04 07:12:32'),(13,'成都车祸','43.4','2022-09-04 07:12:32'),(14,'突然就缓解了我的焦虑','43.3','2022-09-04 07:12:32'),(15,'外婆首次用平板画画惊呆孙子','40.8','2022-09-04 07:12:32'),(16,'浙江三地已宣布停课一天','37.6','2022-09-04 07:12:32'),(17,'上海台风','35.1','2022-09-04 07:12:32'),(18,'女子称自己复活成仙骗男网友486万','29.6','2022-12-04 07:12:32'),(19,'男子回农村花12万买下1200平大院','29.3','2022-12-04 07:12:32'),(20,'网红表演生吃活蜜蜂惨变香肠嘴','22.5','2022-12-04 07:12:32'),(21,'是怎么忍住不跟猫玩的','22.0','2022-12-04 07:12:32'),(22,'12岁男孩玩游戏花掉17万妈妈卖房','21.3','2022-12-04 07:12:32'),(23,'产科医生辟谣双胞胎肚内打架','21.2','2022-12-04 07:12:32');
/*!40000 ALTER TABLE `hotseacher` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2022-12-09  8:55:37
