#include "lenet_inference.h"
#include <stdbool.h>
#include <stdio.h> // Para printf (será substituído por printf)
#include <stdint.h>
#include <string.h>
// #include <math.h> // Não é mais necessário se my_sqrt é usada

#define HW_CONV 0 // Define se a convolução é feita em hardware (1) ou software (0)


// Variável global para o dispositivo ADC (necessária se 'adc' não for globalmente visível para este arquivo)
extern const struct device *const adc; // Declarar como extern para acessar 'adc' do main.c

// Fator Q (definido em main.c)
extern const int Q_FACTOR; 

// Funções auxiliares para obter dimensões de arrays (mantidas)
//#define GET_ARRAY_LEN(array) (sizeof(array) / sizeof(array[0]))
//#define GET_ARRAY_COUNT(array) (sizeof(array) / sizeof(lenet_data_t)) // Ajustado para lenet_data_t


// Printa resultados intermediarios após processamento das camadas
#define PRINTLAYERS 1

// Função de ativação ReLU (opera em ponto fixo)
lenet_data_t relu(lenet_data_t x)
{
  return x * (x > 0);
}

// Implementação de sqrt (ainda opera em double, mas o input/output será escalado)
double my_sqrt(double x)
{
    if (x < 0) return -1; // Erro para números negativos
    if (x == 0) return 0;
    
    double guess = x / 2.0; // Estimativa inicial
    double prev_guess;
    
    for (int i = 0; i < 10; i++) {
        prev_guess = guess;
        guess = (guess + x / guess) / 2.0;
        if (guess == prev_guess) break;
    }
    return guess;
}

// ============================================================================
// Funções de Convolução e Pooling (Substituindo as Macros)
// ============================================================================

// Equivalente a CONVOLUTE_VALID
//void do_convolution_c(uint8_t filter_row_size, uint8_t ofmap_row_size, conv_word_t (*filter_matrix)[MAX_FILTER_ROW_SIZE], conv_word_t (*ifmap_matrix)[MAX_IFMAP_ROW_SIZE], conv_word_t (*expected_ofmap_matrix)[MAX_OFMAP_ROW_SIZE]) {

void convolute_valid(uint8_t filter_row_size, uint8_t ofmap_row_size, lenet_data_t input_arr[LENGTH_FEATURE0][LENGTH_FEATURE0], lenet_data_t output_arr[LENGTH_FEATURE1][LENGTH_FEATURE1], lenet_data_t weight_arr[LENGTH_KERNEL][LENGTH_KERNEL])
{
    //printf("Convoluting with filter GET_ARRAY_LEN(output_arr)=%d GET_ARRAY_LEN(output_arr[0])=%d GET_ARRAY_LEN(weight_arr)=%d GET_ARRAY_LEN(weight_arr[0])=%d...\n", GET_ARRAY_LEN(output_arr), GET_ARRAY_LEN(output_arr[0]), GET_ARRAY_LEN(weight_arr), GET_ARRAY_LEN(weight_arr[0]));
    for (int o0 = 0; o0 < ofmap_row_size; ++o0)
        for (int o1 = 0; o1 < ofmap_row_size; ++o1) {
            int32_t sum_32 = 0; // Usar int32_t para acumulação para evitar overflow
            for (int w0 = 0; w0 < filter_row_size; ++w0)
                for (int w1 = 0; w1 < filter_row_size; ++w1) {
                    // Multiplicação de ponto fixo: (A * B) >> Q_FACTOR
                    sum_32 += ((int32_t)input_arr[o0 + w0][o1 + w1] * weight_arr[w0][w1]);
                    //printf("o0=%d o1=%d w0=%d w1=%d\n", o0, o1, w0, w1);
                }
            output_arr[o0][o1] = (lenet_data_t)(sum_32 >> Q_FACTOR); // Desloca para o formato Q original
        }
    // void do_convolution_c(uint8_t filter_row_size, uint8_t ofmap_row_size, conv_word_t (*filter_matrix)[MAX_FILTER_ROW_SIZE], conv_word_t (*ifmap_matrix)[MAX_IFMAP_ROW_SIZE], conv_word_t (*expected_ofmap_matrix)[MAX_OFMAP_ROW_SIZE]) {
    // for (int i = 0; i < ofmap_row_size; i++) {
    //     for (int j = 0; j < ofmap_row_size; j++) {
    //         int32_t sum = 0;
    //         for (int ii = 0; ii < filter_row_size; ii++) {
    //             for (int jj = 0; jj < filter_row_size; jj++) {
    //                 sum += (int32_t)filter_matrix[ii][jj] * ifmap_matrix[i + ii][j + jj];
    //             }
    //         }
    //         expected_ofmap_matrix[i][j] = (conv_word_t)sum;
    //     }
    // }

}

// Equivalente a SUBSAMP_MAX_FORWARD
void subsamp_max_forward(lenet_data_t input_arr[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1], lenet_data_t output_arr[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2])
{
    //const int len0_ratio = GET_ARRAY_LEN(input_arr[0]) / GET_ARRAY_LEN(output_arr[0]);
    const int len0_ratio = LENGTH_FEATURE1 / LENGTH_FEATURE2;
    //const int len1_ratio = GET_ARRAY_LEN(input_arr[0][0]) / GET_ARRAY_LEN(output_arr[0][0]);
    const int len1_ratio = LENGTH_FEATURE1 / LENGTH_FEATURE2;

    // for (int i = 0; i < GET_ARRAY_LEN(output_arr); ++i)
    //     for (int o0 = 0; o0 < GET_ARRAY_LEN(output_arr[0]); ++o0)
    //         for (int o1 = 0; o1 < GET_ARRAY_LEN(output_arr[0][0]); ++o1)
    for (int i = 0; i < LAYER2; ++i)
        for (int o0 = 0; o0 < LENGTH_FEATURE2; ++o0)
            for (int o1 = 0; o1 < LENGTH_FEATURE2; ++o1)
            {
                lenet_data_t max_val = input_arr[i][o0 * len0_ratio][o1 * len1_ratio]; // Inicializa com o primeiro elemento da janela

                for (int l0 = 0; l0 < len0_ratio; ++l0)
                    for (int l1 = 0; l1 < len1_ratio; ++l1)
                    {
                        if (input_arr[i][o0 * len0_ratio + l0][o1 * len1_ratio + l1] > max_val)
                        {
                            max_val = input_arr[i][o0 * len0_ratio + l0][o1 * len1_ratio + l1];
                        }
                    }
                output_arr[i][o0][o1] = max_val;
            }
}

// Equivalente a CONVOLUTION_FORWARD para a primeira camada (C1)
void convolution_forward_c1(lenet_data_t input_arr[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0], lenet_data_t output_arr[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1], lenet_data_t weight_arr[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL], lenet_data_t bias_arr[LAYER1], lenet_data_t (*action_func)(lenet_data_t), bool hw_conv)
{
    //memset(output_arr, 0, sizeof(output_arr));
    // Substituído memset por loop manual
    // for (int i = 0; i < GET_ARRAY_LEN(output_arr); ++i) {
    //     for (int j = 0; j < GET_ARRAY_LEN(output_arr[0]); ++j) {
    //         for (int k = 0; k < GET_ARRAY_LEN(output_arr[0][0]); ++k) {
    for (int i = 0; i < LAYER1; ++i) {
        for (int j = 0; j < LENGTH_FEATURE1; ++j) {
            for (int k = 0; k < LENGTH_FEATURE1; ++k) {
                output_arr[i][j][k] = 0;
            }
        }
    }

    for (int x_in = 0; x_in < INPUT; ++x_in)
    {
        for (int y_out = 0; y_out < LAYER1; ++y_out)
        {
            if (!hw_conv){
                //uint32_t cyclesw = read_clk_cycle(adc);
                convolute_valid(LENGTH_KERNEL, LENGTH_FEATURE1, input_arr[x_in], output_arr[y_out], weight_arr[x_in][y_out]);   
                //cyclesw = read_clk_cycle(adc) - cyclesw;
                //printf("\x1B[33m" "SW CONV - n_cycles=%u" "\x1B[0m\n", cyclesw);
            }
            else
            {
                //uint32_t cyclehw = read_clk_cycle(adc);
                //accel_convulation(adc, LENGTH_FEATURE0, LENGTH_KERNEL, LENGTH_FEATURE1, weight_arr[x_in][y_out], input_arr[x_in], output_arr[y_out], true, bias_arr[y_out]);
                //cyclehw = read_clk_cycle(adc) - cyclehw;
                //printf("\x1B[33m" "HW CONV - n_cycles=%u" "\x1B[0m\n", cyclehw);
            }
        }
    }
    if (!hw_conv)
        for (int j = 0; j < LAYER1; ++j)
            //for (int i = 0; i < GET_ARRAY_COUNT(output_arr[j]); ++i)
            for (int i = 0; i < LENGTH_FEATURE1; ++i)
                ((lenet_data_t *)output_arr[j])[i] = action_func(((lenet_data_t *)output_arr[j])[i] + bias_arr[j]);
}

// Equivalente a CONVOLUTION_FORWARD para a terceira camada (C3)
void convolution_forward_c3(lenet_data_t input_arr[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2], lenet_data_t output_arr[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3], lenet_data_t weight_arr[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL], lenet_data_t bias_arr[LAYER3], lenet_data_t (*action_func)(lenet_data_t), bool hw_conv)
{
    //memset(output_arr, 0, sizeof(output_arr));
    // Substituído memset por loop manual
    // for (int i = 0; i < GET_ARRAY_LEN(output_arr); ++i) {
    //     for (int j = 0; j < GET_ARRAY_LEN(output_arr[0]); ++j) {
    //         for (int k = 0; k < GET_ARRAY_LEN(output_arr[0][0]); ++k) {
    for (int i = 0; i < LAYER3; ++i) {
        for (int j = 0; j < LENGTH_FEATURE3; ++j) {
            for (int k = 0; k < LENGTH_FEATURE3; ++k) {
                output_arr[i][j][k] = 0;
            }
        }
    }

    for (int x_in = 0; x_in < LAYER2; ++x_in)
    {
        for (int y_out = 0; y_out < LAYER3; ++y_out)
        {
            if (!hw_conv){
                //uint32_t cyclesw = read_clk_cycle(adc);
                convolute_valid(LENGTH_KERNEL, LENGTH_FEATURE3, input_arr[x_in], output_arr[y_out], weight_arr[x_in][y_out]);
                //cyclesw = read_clk_cycle(adc) - cyclesw;
                //printf("\x1B[33m" "SW CONV - n_cycles=%u" "\x1B[0m\n", cyclesw);
            }
            else {
                //uint32_t cyclehw = read_clk_cycle(adc);
                //accel_convulation(adc, LENGTH_FEATURE2, LENGTH_KERNEL, LENGTH_FEATURE3, weight_arr[x_in][y_out], input_arr[x_in], output_arr[y_out], true, bias_arr[y_out]);
                //cyclehw = read_clk_cycle(adc) - cyclehw;
                //printf("\x1B[33m" "HW CONV - n_cycles=%u" "\x1B[0m\n", cyclehw);
            }
        }
    }
    if (!hw_conv)
        for (int j = 0; j < LAYER3; ++j)
            //for (int i = 0; i < GET_ARRAY_COUNT(output_arr[j]); ++i)
            for (int i = 0; i < LENGTH_FEATURE3; ++i)
                ((lenet_data_t *)output_arr[j])[i] = action_func(((lenet_data_t *)output_arr[j])[i] + bias_arr[j]);
}

// Equivalente a CONVOLUTION_FORWARD para a quinta camada (C5)
void convolution_forward_c5(lenet_data_t input_arr[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4], lenet_data_t output_arr[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], lenet_data_t weight_arr[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL], lenet_data_t bias_arr[LAYER5], lenet_data_t (*action_func)(lenet_data_t), bool hw_conv)
{
    //memset(output_arr, 0, sizeof(output_arr));
    // Substituído memset por loop manual
    // for (int i = 0; i < GET_ARRAY_LEN(output_arr); ++i) {
    //     for (int j = 0; j < GET_ARRAY_LEN(output_arr[0]); ++j) {
    //         for (int k = 0; k < GET_ARRAY_LEN(output_arr[0][0]); ++k) {
    for (int i = 0; i < LAYER5; ++i) {
        for (int j = 0; j < LENGTH_FEATURE5; ++j) {
            for (int k = 0; k < LENGTH_FEATURE5; ++k) {
                output_arr[i][j][k] = 0;
            }
        }
    }

    for (int x_in = 0; x_in < LAYER4; ++x_in)
    {
        for (int y_out = 0; y_out < LAYER5; ++y_out)
        {
            if (!hw_conv){
                //uint32_t cyclesw = read_clk_cycle(adc);
                convolute_valid(LENGTH_KERNEL, LENGTH_FEATURE5, input_arr[x_in], output_arr[y_out], weight_arr[x_in][y_out]);
                //cyclesw = read_clk_cycle(adc) - cyclesw;
                //printf("\x1B[33m" "SW CONV - n_cycles=%u" "\x1B[0m\n", cyclesw);
            } else {
            //     //uint32_t cyclehw = read_clk_cycle(adc);
                 //accel_convulation(adc, LENGTH_FEATURE4, LENGTH_KERNEL, LENGTH_FEATURE5, weight_arr[x_in][y_out], input_arr[x_in], output_arr[y_out], true, bias_arr[y_out]);
            //     //cyclehw = read_clk_cycle(adc) - cyclehw;
            //     //printf("\x1B[33m" "HW CONV - n_cycles=%u" "\x1B[0m\n", cyclehw);
             }
        }
    }
    if (!hw_conv)
        for (int j = 0; j < LAYER5; ++j)
            //for (int i = 0; i < GET_ARRAY_COUNT(output_arr[j]); ++i)
            for (int i = 0; i < LENGTH_FEATURE5; ++i)
                ((lenet_data_t *)output_arr[j])[i] = action_func(((lenet_data_t *)output_arr[j])[i] + bias_arr[j]);
}


// Equivalente a DOT_PRODUCT_FORWARD
void dot_product_forward(lenet_data_t input_arr[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], lenet_data_t output_arr[OUTPUT], lenet_data_t weight_arr[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT], lenet_data_t bias_arr[OUTPUT], lenet_data_t (*action_func)(lenet_data_t))
{
    const int input_flat_size = LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5;

    //memset(output_arr, 0, sizeof(output_arr));
    for (int i = 0; i < OUTPUT; ++i) { // Itera diretamente sobre o tamanho da dimensão
        output_arr[i] = 0;
    }

    for (int x = 0; x < input_flat_size; ++x)
    {
        for (int y = 0; y < OUTPUT; ++y)
        {
            // Multiplicação de ponto fixo e acumulação
            ((lenet_data_t *)output_arr)[y] += (lenet_data_t)(((int32_t)((lenet_data_t *)input_arr)[x] * weight_arr[x][y]) >> Q_FACTOR);
        }
    }
    //for (int j = 0; j < GET_ARRAY_LEN(bias_arr); ++j)
    for (int j = 0; j < OUTPUT; ++j)
        ((lenet_data_t *)output_arr)[j] = action_func(((lenet_data_t *)output_arr)[j] + bias_arr[j]);
}


// ============================================================================
// Funções Principais da LeNet (mantidas)
// ============================================================================

static void forward(LeNet5 *lenet, Feature *features, lenet_data_t(*action)(lenet_data_t), bool hw_conv)
{

#ifdef PRINTLAYERS
  printf("#############\n");
  for (int i=0; i<32; i++) {
    for (int j=0; j<32; j++) {
        printf("%4d", features->input[0][i][j]);
    }
    printf("\n");
  }
  printf("--------------------\n");
#endif //PRINTLAYERS

  //printf("calling convolution_forward_c1...\n");
  convolution_forward_c1(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action, hw_conv);
  //printf("cycle pos conv for: %u\n", rdcycle());
  
#ifdef PRINTLAYERS
  printf("#############\n");
  for (int k=0; k<6; k++) {
    for (int i=0; i<28; i++) {
        for (int j=0; j<28; j++) {
            printf("%4d", features->layer1[k][i][j]);
        }
        printf("\n");
    }
  }
  printf("--------------------\n");
#endif //PRINTLAYERS


  subsamp_max_forward(features->layer1, features->layer2);
  //printf("cycle pos sub max for: %u\n", rdcycle());

  //printf("calling convolution_forward_c3...\n");
  convolution_forward_c3(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action, hw_conv);
  //printf("cycle pos conv for2 : %u\n", rdcycle());

  subsamp_max_forward(features->layer3, features->layer4);
  //printf("cycle pos sub max for2: %u\n", rdcycle());

  //printf("calling convolution_forward_c5...\n");
  convolution_forward_c5(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action, false); //force to SW conv
  //printf("cycle pos conv for3 : %u\n", rdcycle());

  dot_product_forward(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
  //printf("cycle pos dot  for : %u\n", rdcycle());

}

// ============================================================================
//     PARÂMETROS DE NORMALIZAÇÃO E QUANTIZAÇÃO (OBTIDOS DO PYTHON)
// ===========================================================================
#define MNIST_MEAN         0.1307f
#define MNIST_STD          0.3081f
#define INPUT_SCALE        0.012722019f
#define INPUT_ZERO_POINT   33

/**
 * @brief Arredonda um float para o inteiro mais próximo
 *
 * @param x O valor float a ser arredondado.
 * @return O valor inteiro arredondado.
 */
int32_t custom_round(float x)
{
    if (x >= 0.0f) {
        return (int32_t)(x + 0.5f);
    }
    return (int32_t)(x - 0.5f);
}

/**
 * @brief Normaliza e quantiza a imagem de entrada para o formato esperado pela rede.
 *
 * @param raw_input       Ponteiro para a matriz de entrada 28x28 com valores brutos (0-255).
 * @param processed_input Ponteiro para a matriz de saída 32x32 (com padding) no formato de ponto fixo (int8_t).
 */
void load_input(lenet_data_t processed_input[LENGTH_FEATURE0][LENGTH_FEATURE0],uint8_t raw_input[28][28])
{
    const int padding = 2; // Padding para transformar 28x28 em 32x32
    const float scale_factor = 1.0f / INPUT_SCALE;

    // 1. Limpar a matriz de saída (preencher tudo com o valor quantizado de "fundo")
    float normalized_background = (0.0f - MNIST_MEAN) / MNIST_STD;
    lenet_data_t fixed_point_background = (lenet_data_t)custom_round(normalized_background * scale_factor + INPUT_ZERO_POINT);

    for (int i = 0; i < LENGTH_FEATURE0; i++) {
        for (int j = 0; j < LENGTH_FEATURE0; j++) {
            processed_input[i][j] = fixed_point_background;
        }
    }

    // 2. Processar e inserir a imagem 28x28 na matriz 32x32 com padding
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            // Normalizar para float
            float normalized_pixel = ((float)raw_input[i][j] / 255.0f - MNIST_MEAN) / MNIST_STD;

            // Quantizar para int8_t (ou o tipo de 'lenet_data_t')
            int32_t quant_val = custom_round(normalized_pixel * scale_factor + INPUT_ZERO_POINT);

            // Limitar o valor ao intervalo de int8_t para segurança
            if (quant_val > 127) quant_val = 127;
            if (quant_val < -128) quant_val = -128;

            processed_input[i + padding][j + padding] = (lenet_data_t)quant_val;
        }
    }
}

uint8 get_result(Feature *features, uint8 count)
{
  lenet_data_t *output = (lenet_data_t *)features->output; 
  uint8 result = 0;
  lenet_data_t maxvalue = output[0]; // Inicializa com o primeiro valor
  
#ifdef PRINTLAYERS
  printf("\nOutput scores:\n");
    for(uint8 i=0; i<count; i++) {
        printf("Class %d: %6d\n", i, ((lenet_data_t*)features->output)[i]);
    }
#endif // PRINTLAYERS

  for (uint8 i = 1; i < count; ++i)
  {
    if (output[i] > maxvalue)
    {
      maxvalue = output[i];
      result = i;
    }
  }
  return result;
}


void print_input(image input) {

  for (int i=0; i<28; i++) {
    for (int j=0; j<28; j++) {
        printf("%4d", input[i][j]);
    }
    printf("\n");
  }

}
uint8 Predict(LeNet5 *lenet, image input, uint8 count)
{
  //printf("Predict...\n");
  static Feature features = { 0 }; // Inicializa todos os membros da struct com 0

#ifdef PRINTLAYERS
  print_input(input);
#endif // PRINTLAYERS

 
  load_input(&features, input);


  forward(lenet, &features, relu, HW_CONV);

  uint8 result = get_result(&features, count);
  
  return result;
}
