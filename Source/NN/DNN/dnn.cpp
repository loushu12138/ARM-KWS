

#include "dnn.h"
//float gruou[5];

const float gate_weights[26*32]=GATE_WEIGHTS;
const float gate_bias[32]=GATE_BIAS;
const float candidate_weights[26*16]=CANDIDATE_WEIGHTS;
const float candidate_bias[16]=CANDIDATE_BIAS;
const float last_weights[16*5]=LAST_WEIGHTS;
const float last_bias[5]=LAST_BIAS;

void gru(const float *gate_weights, const float *gate_bias,

         const float *candidate_weights, const float *candidate_bias,

         float *data, float *output_data) {

  float state[16];

  for (int i = 0; i < 16; i++) {

    state[i] = 0;

  }

  float r_t[16];

  float z_t[16];

  float h_t[16];

  for (int l = 0; l < 25; l++) {

    for (int o = 0; o < 16; o++) {

      r_t[o] = 0;

      z_t[o] = 0;

      for (int i = 0; i < 16; i++) {

        r_t[o] += state[i] * gate_weights[(i + 10) * 2 * 16 + o];

        z_t[o] += state[i] * gate_weights[(i + 10) * 2 * 16 + 16 + o];

      }
    
    //   if (l==3) {
    //    for (int i =0;i<5;i++){
    //    gruou[i] = z_t[i];
    //   }
    //  }
      

      for (int i = 0; i < 10; i++) {

        r_t[o] += data[i + l * 10] * gate_weights[i * 2 * 16 + o];

        z_t[o] += data[i + l * 10] * gate_weights[i * 2 * 16 + 16 + o];

      }//lianghua2
      // if (l==0) {
      // for (int i =0;i<5;i++){
       //gruou[i] = z_t[i];
      //}
      //}

      r_t[o] = 1. / (1. + exp(-(r_t[o] + gate_bias[o])));

      z_t[o] = 1. / (1. + exp(-(z_t[o] + gate_bias[16 + o])));
      // if (l==1) {
      //   for (int i =0;i<5;i++){
      //   gruou[i] = z_t[i];
      //  }
      // }
    }
    // if (l==13) {
    //    for (int i =0;i<5;i++){
    //    gruou[i] =z_t[i]; //r_t[i];
    //   }
    //  }

    for (int o = 0; o < 16; o++) {

      h_t[o] = 0;

      for (int i = 0; i < 16; i++) {

        h_t[o] += state[i] * r_t[i] * candidate_weights[(i + 10) * 16 + o];//lianghua3

      }

      for (int i = 0; i < 10; i++) {

        h_t[o] += data[i + l * 10] * candidate_weights[i * 16 + o];//lianghua4

      }

      h_t[o] += candidate_bias[o];//lianghua5

      h_t[o] = (1. - exp(-2 * h_t[o])) / (1. + exp(-2 * h_t[o]));

    }
    // if (l==2) {
    //    for (int i =0;i<5;i++){
    //    gruou[i] =state[i]; //r_t[i];
    //   }
    //  }

    for (int o = 0; o < 16; o++) {

      state[o] = z_t[o] * state[o] + (1 - z_t[o]) * h_t[o];//lianghua6

    //   if (l==24) {
    //   for (int i =0;i<5;i++){
    //    gruou[i] =state[i]; //r_t[i];
    //   }
    //  }

      //output_data[o + l * 16] = state[o];

    }
      
  }
  for (int o = 0; o < 16; o++) {
       output_data[o] = state[o];}
}

DNN::DNN()
{
  
  frame_len = FRAME_LEN;
  frame_shift = FRAME_SHIFT;
  num_mfcc_features = NUM_MFCC_COEFFS;
  num_frames = NUM_FRAMES;
  num_out_classes = OUT_DIM;
 
}



void DNN::run_nn (float* in_data, float* out_data)
{  //for (int i =0;i<5;i++){
     //  gruou[i] = in_data[i];}
 
  
  
  float output_data[16];
  gru(gate_weights, gate_bias, candidate_weights, candidate_bias, in_data, output_data);

   for (int o=0; o<5; o++){
     out_data[o]=0;
     for(int i=0; i<16; i++){
       out_data[o] +=output_data[i]*last_weights[i*5+o];
     }
     out_data[o] +=last_bias[o];
   }
// for (int i =0;i<5;i++){
      // gruou[i] = out_data[i];}
 
 }


