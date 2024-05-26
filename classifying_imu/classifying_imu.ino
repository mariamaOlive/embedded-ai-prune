/*
  IMU Classifier
  This example uses the on-board IMU to start reading acceleration and gyroscope
  data from on-board IMU, once enough samples are read, it then uses a
  TensorFlow Lite (Micro) model to try to classify the movement as a known gesture.
  Note: The direct use of C/C++ pointers, namespaces, and dynamic memory is generally
        discouraged in Arduino examples, and in the future the TensorFlowLite library
        might change to make the sketch simpler.
  The circuit:
  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.
  Created by Don Coleman, Sandeep Mistry
  Modified by Dominic Pajak, Sandeep Mistry
  This example code is in the public domain.
*/

// #include "Arduino_LSM9DS1.h"

#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "output_files/sample_0.h"

#include "content/model.h"

const float accelerationThreshold = 2.5; // threshold of significant in G's
const int numSamples = 119;

int samplesRead = numSamples;

// global variables used for TensorFlow Lite (Micro)
// tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 180 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* GESTURES[] = {
  "punch",
  "flex"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println();

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  Serial.println("Model Type:");
  // Serial.println(tflModel->type());

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
    for (int i=0; i<28*28; i++){
      // input the unsigned int data from the header
      tflInputTensor->data.int8[i] = static_cast<uint8_t>(image_data[i] / 255.0f * 255); // Normalize and scale to 0-255
    }

      // Run inferencing
    TfLiteStatus invokeStatus = tflInterpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
      Serial.println("Invoke failed!");
      return;
    }
    
    // // Process and print the output
    tflOutputTensor = tflInterpreter->output(0);
    float scale = tflOutputTensor->params.scale;
    int zero_point = tflOutputTensor->params.zero_point;

    uint8_t* probabilities = tflOutputTensor->data.uint8;

  float max_probability = -1.0;  // Initialize to a value lower than the minimum possible probability
  int max_index = -1;  // Initialize to an invalid index

  for (int i = 0; i < 10; i++) {
    Serial.print(": ");
    float probability = (probabilities[i] - zero_point) * scale;
    Serial.print("Class ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(probability, 6);

    // Check if this probability is the highest we've seen so far
    if (probability > max_probability) {
        max_probability = probability;
        max_index = i;
    }
  }

    // Print the class with the largest probability
    Serial.print("Class with largest probability: ");
    Serial.print("Class ");
    Serial.print(max_index);
    Serial.print(": ");
    Serial.println(max_probability, 6);

    Serial.println();
    delay(10000);
}