/*
@author : 范文捷
@data    : 2016-04-20
@note   : 根据Yann Lecun的论文《Gradient-based Learning Applied To Document Recognition》编写
@api    :

// Funções de API (mantidas)
// void TrainBatch(LeNet5 *lenet, image *inputs, const char(*resMat)[OUTPUT],uint8 *labels, int batchSize);
// void Train(LeNet5 *lenet, image input, const char(*resMat)[OUTPUT],uint8 label);
uint8 Predict(LeNet5 *lenet, image input, uint8 count);
// void Initial(LeNet5 *lenet);
*/

#pragma once

#include <stdint.h> // Para int8_t, int16_t, int32_t
#include <stdbool.h>

// Definições de constantes (mantidas)
#define LENGTH_KERNEL   5

#define LENGTH_FEATURE0 32
#define LENGTH_FEATURE1 (LENGTH_FEATURE0 - LENGTH_KERNEL + 1) // 32 - 5 + 1 = 28
#define LENGTH_FEATURE2 (LENGTH_FEATURE1 >> 1) // 28 / 2 = 14
#define LENGTH_FEATURE3 (LENGTH_FEATURE2 - LENGTH_KERNEL + 1) // 14 - 5 + 1 = 10
#define LENGTH_FEATURE4 (LENGTH_FEATURE3 >> 1) // 10 / 2 = 5
#define LENGTH_FEATURE5 (LENGTH_FEATURE4 - LENGTH_KERNEL + 1) // 5 - 5 + 1 = 1

#define INPUT           1
#define LAYER1          6
#define LAYER2          6
#define LAYER3          16
#define LAYER4          16
#define LAYER5          120
#define OUTPUT          10

#define ALPHA 0.5
#define PADDING 2

// Typedefs (mantidos)
typedef unsigned char uint8;
typedef uint8 image[28][28]; // Imagem de entrada 28x28

// Tipo de dados parametrizável para os pesos e feature maps da LeNet
// Será definido em main.c (ex: int8_t)
typedef int8_t lenet_data_t; 
typedef int32_t lenet_bias_t; 

// Fator Q para ponto fixo (definido em main.c)
extern const int Q_FACTOR; 


// Estruturas (alteradas para usar lenet_data_t e lenet_bias_t)
typedef struct LeNet5
{
    // Pesos e biases para cada camada
    // As camadas de peso terão sempre valores de 8 bits com sinal
    lenet_data_t weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL]; // Conv1: 1 entrada, 6 filtros 5x5
    lenet_data_t weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL]; // Conv3: 6 entradas, 16 filtros 5x5
    lenet_data_t weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL]; // Conv5: 16 entradas, 120 filtros 5x5
    lenet_data_t weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT]; // Camada totalmente conectada

    // As camadas de peso serão sempre de 32 bits com sinal
    lenet_bias_t bias0_1[LAYER1];
    lenet_bias_t bias2_3[LAYER3];
    lenet_bias_t bias4_5[LAYER5];
    lenet_bias_t bias5_6[OUTPUT];

} LeNet5;

typedef struct Feature
{
    // Feature maps para cada camada
    // As entradas e saídas dos features maps serão sempre de 8 bits
    lenet_data_t input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0]; // Entrada com padding (32x32)
    lenet_data_t layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1]; // Saída Conv1 (28x28)
    lenet_data_t layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2]; // Saída Subsampling2 (14x14)
    lenet_data_t layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3]; // Saída Conv3 (10x10)
    lenet_data_t layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4]; // Saída Subsampling4 (5x5)
    lenet_data_t layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5]; // Saída Conv5 (1x1)
    lenet_data_t output[OUTPUT]; // Saída da camada final (10 elementos)
} Feature;

void print_input(image input);

// Funções de ativação e utilitários
lenet_data_t relu(lenet_data_t x);
double my_sqrt(double x);
int32_t custom_round(float x);

// Funções principais da rede
uint8 Predict(LeNet5 *lenet, image input, uint8 count);
void load_input(lenet_data_t processed_input[32][32], uint8_t raw_input[28][28]);
uint8 get_result(Feature *features, uint8 count);

// Operações de rede
void convolution_forward_c1(lenet_data_t input[1][32][32], lenet_data_t output[6][28][28],
                            lenet_data_t weights[1][6][5][5], lenet_data_t bias[6],
                            lenet_data_t (*action)(lenet_data_t), bool hw);

void convolution_forward_c3(lenet_data_t input[6][14][14], lenet_data_t output[16][10][10],
                            lenet_data_t weights[6][16][5][5], lenet_data_t bias[16],
                            lenet_data_t (*action)(lenet_data_t), bool hw);

void convolution_forward_c5(lenet_data_t input[16][5][5], lenet_data_t output[120][1][1],
                            lenet_data_t weights[16][120][5][5], lenet_data_t bias[120],
                            lenet_data_t (*action)(lenet_data_t), bool hw);

void subsamp_max_forward(lenet_data_t input[6][28][28], lenet_data_t output[6][14][14]);
void subsamp_max_forward_2(lenet_data_t input[16][10][10], lenet_data_t output[16][5][5]); // se desejar separar

void dot_product_forward(lenet_data_t input[120][1][1], lenet_data_t output[10],
                         lenet_data_t weight[120][10], lenet_data_t bias[10],
                         lenet_data_t (*action)(lenet_data_t));

// Funções auxiliares
void convolute_valid(uint8_t filter_size, uint8_t ofmap_size,
                     lenet_data_t input[32][32], lenet_data_t output[28][28],
                     lenet_data_t kernel[5][5]);
