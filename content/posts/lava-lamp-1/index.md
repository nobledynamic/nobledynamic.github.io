---
title: "Lava Lamp 1"
summary: "TBD"
date: 2024-04-19T13:00:43Z
draft: true
showAuthor: true
authors:
  - "rogernoble"
tags:
  - "Microsoft Fabric"
  - "Image Processing"
series: ["Lava Lamp"]
series_order: 1
---

I recently came across the Cloudflare blog post on how they use [Lava Lamps](https://blog.cloudflare.com/randomness-101-lavarand-in-production) to generate random numbers and was fascinated by the idea. For those who are not familiar with the concept, the idea is to capture the random motion of the wax in a lava lamp and use this as a source of entropy for generating random numbers. The randomness is captured by a camera and then processed to generate random numbers.

The idea is that the motion of the wax is unpredictable and therefore can be used as a source of randomness. I thought it would be fun to try and replicate this using Microsoft Fabric. This post will cover the first part of the project, which is to create a simple Lava Lamp image using Fabric.

# Background

As detailed in the post [LavaRand in Production: The Nitty-Gritty Technical Details](https://blog.cloudflare.com/lavarand-in-production-the-nitty-gritty-technical-details), Cloudflare have a wall of lava lamps in their office that they use to generate random numbers. Randomness is an important part of security and cryptography, and having a good source of randomness is crucial. Computers are deterministic machines, and generating truly random numbers is difficult, this means that asking a computer to generate a random number will always result in a pseudo-random number. This also means that asking for a random number twice will result in the same number being generated.

This problem is overcome by introducing what's know as entropy, which is used as the starting point for generating random numbers. A naive approach would be to use the current time as the source of entropy, but this is not a good idea as the time can be predicted. Instead, a better approach is to use a truly random source of entropy. Lucky for us unpredicatable sources abound in the physical world, which is where the lava lamps come in, as the motion of the wax is unpredictable, it can be used as a source of entropy to generate random numbers.

# Images

Images are a fantastic way to generate randomness and can quite easily be turned into very large numbers. When an image is captured, each pixel contains measurements of the amount of red, green, and blue light that was detected by the sensor at that point. In a standard 8 bit image (the bit depth used by most consumer cameras), each of these three values is represented by a number between 0 and 255. This means that an image can be thought of as a very large number, where each pixel is a digit in the number. The idea is to take an image of a lava lamp and then process this image to generate random numbers.

# The setup

First I need a source of images. I happened to have a lava lamp at home, so I set this up in a dark room and took a photo.

I create a new Storage Account in Azure and upload the image to a Blob Storage container. I can then head over to Fabric, create a new workspace, a lakehouse and then set up a shortcut to the image in the Blob Storage container.

# The code

With a new notebook open in Fabric, I can start writing some code to load the image and display it.

```python