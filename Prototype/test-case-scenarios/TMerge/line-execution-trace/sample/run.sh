cd ..
gradle shadowJar
cd sample
cp ../build/libs/shadow.jar ./ 
javac Main.java Calculator.java -d ./
java -javaagent:shadow.jar="-c main/Main;main/Calculator" main.Main -classpath ./ 
rm -rf shadow.jar
