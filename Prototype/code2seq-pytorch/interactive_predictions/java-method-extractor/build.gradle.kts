import org.gradle.jvm.tasks.Jar

plugins {
    kotlin("jvm") version "1.3.72"

    // Apply the application plugin to add support for building a CLI application.
    application

    id("com.github.johnrengelman.shadow") version "5.0.0"
}


group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib-jdk8"))
    implementation("com.github.javaparser:javaparser-symbol-solver-core:3.15.17")
    implementation("com.xenomachina:kotlin-argparser:2.0.7")
}

tasks {
    compileKotlin {
        kotlinOptions.jvmTarget = "1.8"
    }
    compileTestKotlin {
        kotlinOptions.jvmTarget = "1.8"
    }
    shadowJar {
        // defaults to project.name
        //archiveBaseName.set("${project.name}-fat")

        // defaults to all, so removing this overrides the normal, non-fat jar
        archiveClassifier.set("")
    }
}

application {
    // Define the main class for the application.
    mainClassName = "method.extractor.AppKt"
}
