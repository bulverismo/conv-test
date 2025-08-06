#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define GETLENGTH_FIXED(array) (sizeof(array) / sizeof(*(array)))
#define GETDIM2(array)   (sizeof(array[0]) / sizeof(*(array[0])))
#define GETDIM3(array)   (sizeof(array[0][0]) / sizeof(*(array[0][0])))
#define FOREACH_FIXED(i, count) for (int i = 0; i < (int)(count); ++i)
#define FIXED_MUL(a, b) ((fixed)(((int32_t)(a) * (b)) >> FIXED_SCALE))


#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))
#define GETCOUNT(array)  (sizeof(array)/sizeof(double))
#define FOREACH(i,count) for (int i = 0; i < count; ++i)



#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}

#define CONVOLUTE_VALID_FIXED(input, output, weight) { \
    FOREACH_FIXED(o0, GETLENGTH_FIXED(output)) \
    FOREACH_FIXED(o1, GETLENGTH_FIXED(*(output))) { \
        int64_t acc = 0; \
        FOREACH_FIXED(w0, GETLENGTH_FIXED(weight)) \
        FOREACH_FIXED(w1, GETLENGTH_FIXED(*(weight))) { \
            acc += (int64_t)input[o0 + w0][o1 + w1] * weight[w0][w1]; \
        } \
        output[o0][o1] = (fixed)(acc >> FIXED_SCALE); \
    } \
}

// Função ReLU para ponto fixo
fixed relu_fixed(fixed x) {
    return (x > 0) ? x : 0;
}



static void forward_fixed(LeNet5_fixed *lenet_fixed, Feature_fixed *features_fixed) {
    // Conv1: input 32x32 -> 28x28
    for (int oc = 0; oc < LAYER1; oc++) {
        for (int y = 0; y < LENGTH_FEATURE1; y++) {
            for (int x = 0; x < LENGTH_FEATURE1; x++) {
                int32_t sum = 0;
                for (int ic = 0; ic < INPUT; ic++) {
                    for (int ky = 0; ky < LENGTH_KERNEL; ky++) {
                        for (int kx = 0; kx < LENGTH_KERNEL; kx++) {
                            sum += (int32_t)features_fixed->input[ic][y + ky][x + kx] * 
                                   lenet_fixed->weight0_1[ic][oc][ky][kx];
                        }
                    }
                }
                features_fixed->layer1[oc][y][x] = (fixed)(sum >> FIXED_SCALE) + 
                                                   lenet_fixed->bias0_1[oc];
                // ReLU
                if (features_fixed->layer1[oc][y][x] < 0) {
                    features_fixed->layer1[oc][y][x] = 0;
                }
            }
        }
    }

    // Max Pooling 1: 28x28 -> 14x14
    const int pool_size1 = 2;
    for (int c = 0; c < LAYER1; c++) {
        for (int y = 0; y < LENGTH_FEATURE2; y++) {
            for (int x = 0; x < LENGTH_FEATURE2; x++) {
                fixed max_val = features_fixed->layer1[c][y * pool_size1][x * pool_size1];
                for (int py = 0; py < pool_size1; py++) {
                    for (int px = 0; px < pool_size1; px++) {
                        fixed val = features_fixed->layer1[c][y * pool_size1 + py][x * pool_size1 + px];
                        if (val > max_val) max_val = val;
                    }
                }
                features_fixed->layer2[c][y][x] = max_val;
            }
        }
    }

    // Conv2: 14x14 -> 10x10
    for (int oc = 0; oc < LAYER3; oc++) {
        for (int y = 0; y < LENGTH_FEATURE3; y++) {
            for (int x = 0; x < LENGTH_FEATURE3; x++) {
                int32_t sum = 0;
                for (int ic = 0; ic < LAYER2; ic++) {
                    for (int ky = 0; ky < LENGTH_KERNEL; ky++) {
                        for (int kx = 0; kx < LENGTH_KERNEL; kx++) {
                            sum += (int32_t)features_fixed->layer2[ic][y + ky][x + kx] * 
                                   lenet_fixed->weight2_3[ic][oc][ky][kx];
                        }
                    }
                }
                features_fixed->layer3[oc][y][x] = (fixed)(sum >> FIXED_SCALE) + 
                                                   lenet_fixed->bias2_3[oc];
                // ReLU
                if (features_fixed->layer3[oc][y][x] < 0) {
                    features_fixed->layer3[oc][y][x] = 0;
                }
            }
        }
    }

    // Max Pooling 2: 10x10 -> 5x5
    const int pool_size2 = 2;
    for (int c = 0; c < LAYER3; c++) {
        for (int y = 0; y < LENGTH_FEATURE4; y++) {
            for (int x = 0; x < LENGTH_FEATURE4; x++) {
                fixed max_val = features_fixed->layer3[c][y * pool_size2][x * pool_size2];
                for (int py = 0; py < pool_size2; py++) {
                    for (int px = 0; px < pool_size2; px++) {
                        fixed val = features_fixed->layer3[c][y * pool_size2 + py][x * pool_size2 + px];
                        if (val > max_val) max_val = val;
                    }
                }
                features_fixed->layer4[c][y][x] = max_val;
            }
        }
    }

    // Conv3: 5x5 -> 1x1
    for (int oc = 0; oc < LAYER5; oc++) {
        for (int y = 0; y < LENGTH_FEATURE5; y++) {
            for (int x = 0; x < LENGTH_FEATURE5; x++) {
                int32_t sum = 0;
                for (int ic = 0; ic < LAYER4; ic++) {
                    for (int ky = 0; ky < LENGTH_KERNEL; ky++) {
                        for (int kx = 0; kx < LENGTH_KERNEL; kx++) {
                            sum += (int32_t)features_fixed->layer4[ic][y + ky][x + kx] * 
                                   lenet_fixed->weight4_5[ic][oc][ky][kx];
                        }
                    }
                }
                features_fixed->layer5[oc][y][x] = (fixed)(sum >> FIXED_SCALE) + 
                                                   lenet_fixed->bias4_5[oc];
                // ReLU
                if (features_fixed->layer5[oc][y][x] < 0) {
                    features_fixed->layer5[oc][y][x] = 0;
                }
            }
        }
    }

    // Flatten
    fixed flattened[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5];
    int index = 0;
    for (int c = 0; c < LAYER5; c++) {
        for (int y = 0; y < LENGTH_FEATURE5; y++) {
            for (int x = 0; x < LENGTH_FEATURE5; x++) {
                flattened[index++] = features_fixed->layer5[c][y][x];
            }
        }
    }

    // Fully Connected
    for (int i = 0; i < OUTPUT; i++) {
        int32_t sum = 0;
        for (int j = 0; j < LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5; j++) {
            sum += (int32_t)flattened[j] * lenet_fixed->weight5_6[j][i];
        }
        features_fixed->output[i] = (fixed)(sum >> FIXED_SCALE) + 
                                    lenet_fixed->bias5_6[i];
    }
}

static inline void fully_connected_fixed(fixed input[], 
                                       fixed output[], 
                                       fixed weights[][OUTPUT], 
                                       fixed bias[],
                                       int input_size, 
                                       int output_size) {
    for (int i = 0; i < output_size; i++) {
        int64_t sum = 0;
        for (int j = 0; j < input_size; j++) {
            sum += (int64_t)input[j] * weights[j][i];
        }
        output[i] = (fixed)((sum >> FIXED_SCALE) + bias[i]);
    }
}



#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}

#define CONVOLUTE_FULL_FIXED(input, output, weight) { \
    FOREACH_FIXED(i0, GETLENGTH_FIXED(input)) \
      FOREACH_FIXED(i1, GETLENGTH_FIXED(*(input))) \
        FOREACH_FIXED(w0, GETLENGTH_FIXED(weight)) \
          FOREACH_FIXED(w1, GETLENGTH_FIXED(*(weight))) { \
            int64_t product = (int64_t)(input)[i0][i1] * \
                              (int64_t)(weight)[w0][w1]; \
            (output)[i0 + w0][i1 + w1] += (fixed)(product >> FIXED_SCALE); \
          } \
}

#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
}


// Substitua a macro CONVOLUTION_FORWARD_FIXED por:
#define CONVOLUTION_FORWARD_FIXED(input, output, weight, bias, action) { \
    for (int x = 0; x < GETLENGTH_FIXED(weight); ++x) { \
        for (int y = 0; y < GETLENGTH_FIXED(*weight); ++y) { \
            CONVOLUTE_VALID_FIXED(input[x], output[y], weight[x][y]); \
        } \
    } \
    FOREACH_FIXED(j, GETLENGTH_FIXED(output)) { \
        const int rows = GETLENGTH_FIXED(output[j]); \
        const int cols = GETLENGTH_FIXED(output[j][0]); \
        for (int r = 0; r < rows; r++) { \
            for (int c = 0; c < cols; c++) { \
                fixed temp = output[j][r][c] + bias[j]; \
                output[j][r][c] = action(temp); \
            } \
        } \
    } \
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
}


#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}

// Substitua a macro original por:
#define SUBSAMP_MAX_FORWARD_FIXED(input, output) { \
    const int len0 = GETLENGTH_FIXED(*(input)) / GETLENGTH_FIXED(*(output)); \
    const int len1 = GETLENGTH_FIXED(**(input)) / GETLENGTH_FIXED(**(output)); \
    FOREACH_FIXED(i, GETLENGTH_FIXED(output)) \
    FOREACH_FIXED(o0, GETLENGTH_FIXED(*(output))) \
    FOREACH_FIXED(o1, GETLENGTH_FIXED(**(output))) { \
        fixed max_val = input[i][o0 * len0][o1 * len1]; \
        FOREACH_FIXED(l0, len0) \
        FOREACH_FIXED(l1, len1) { \
            fixed val = input[i][o0 * len0 + l0][o1 * len1 + l1]; \
            if (val > max_val) max_val = val; \
        } \
        output[i][o0][o1] = max_val; \
    } \
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
	FOREACH(i, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH(**(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];							\
	}																							\
}

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y];	\
	FOREACH(j, GETLENGTH(bias))												\
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);	\
}

#define DOT_PRODUCT_FORWARD_FIXED(input, output, weight, bias, action) { \
    const int input_size = LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5; \
    for (int y = 0; y < OUTPUT; y++) { \
        int64_t sum = 0; \
        for (int x = 0; x < input_size; x++) { \
            sum += (int64_t)((fixed *)input)[x] * weight[x][y]; \
        } \
        output[y] = action((fixed)(sum >> FIXED_SCALE) + bias[y]); \
    } \
}

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
	FOREACH(i, GETCOUNT(inerror))												\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((double *)outerror)[j];										\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
}

double relu(double x)
{
	return x*(x > 0);
}

fixed identity_fixed(fixed x) {
    return x;
}

double relugrad(double y)
{
	return y > 0;
}

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{
	CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}


static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}

static inline void load_input(Feature *features, image input)
{
	double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
	const long sz = sizeof(image) / sizeof(**input);
	double mean = 0, std = 0;
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}
}

static inline void load_input_fixed(Feature_fixed *features, image input) {
    // Cálculo idêntico à versão float
    double mean = 0, std = 0;
    const int pixels = 28*28;
    
    for (int j = 0; j < 28; j++) {
        for (int k = 0; k < 28; k++) {
            mean += input[j][k];
            std += input[j][k] * input[j][k];
        }
    }
    mean /= pixels;
    std = sqrt(std / pixels - mean*mean);
    
    // Preencher com padding
    for (int j = 0; j < LENGTH_FEATURE0; j++) {
        for (int k = 0; k < LENGTH_FEATURE0; k++) {
            features->input[0][j][k] = 0;
        }
    }
    
    // Aplicar mesma normalização que a versão float
    for (int j = 0; j < 28; j++) {
        for (int k = 0; k < 28; k++) {
            double normalized = (input[j][k] - mean) / std;
            features->input[0][j + PADDING][k + PADDING] = FLOAT_TO_FIXED(normalized);
        }
    }
}


static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
	double inner = 0;
	for (int i = 0; i < count; ++i)
	{
		double res = 0;
		for (int j = 0; j < count; ++j)
		{
			res += exp(input[j] - input[i]);
		}
		loss[i] = 1. / res;
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];
	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;
	}
}

static void load_target(Feature *features, Feature *errors, int label)
{
	double *output = (double *)features->output;
	double *error = (double *)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
{
	double *output = (double *)features->output; 
	uint8 result = 0;
	double maxvalue = *output;
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

static uint8 get_result_fixed(Feature_fixed *features_fixed, uint8 count) {
    fixed *output = features_fixed->output; 
    uint8 result = 0;
    fixed maxvalue = output[0];
    for (uint8 i = 1; i < count; ++i) {
        if (output[i] > maxvalue) {
            maxvalue = output[i];
            result = i;
        }
    }
    return result;
}
static double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		srand((unsigned)time(0));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	return *(double *)&lvalue - 3;
}


void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize)
{
	double buffer[GETCOUNT(LeNet5)] = { 0 };
	int i = 0;
	for (i = 0; i < batchSize; ++i)
	{
		Feature features = { 0 };
		Feature errors = { 0 };
		LeNet5	deltas = { 0 };
		load_input(&features, inputs[i]);
		forward(lenet, &features, relu);
		load_target(&features, &errors, labels[i]);
		backward(lenet, &deltas, &errors, &features, relugrad);
		{
			FOREACH(j, GETCOUNT(LeNet5))
				buffer[j] += ((double *)&deltas)[j];
		}
	}
	double k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += k * buffer[i];
}

void Train(LeNet5 *lenet, image input, uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	LeNet5 deltas = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	load_target(&features, &errors, label);
	backward(lenet, &deltas, &errors, &features, relugrad);
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += ALPHA * ((double *)&deltas)[i];
}

uint8 Predict_fixed(LeNet5_fixed *lenet_fixed, image input,uint8 count)
{
	Feature_fixed features_fixed = { 0 };
	load_input_fixed(&features_fixed, input);
	forward_fixed(lenet_fixed, &features_fixed);
	return get_result_fixed(&features_fixed, count);
}

uint8 Predict(LeNet5 *lenet, image input,uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	return get_result(&features, count);
}

void Initial(LeNet5 *lenet)
{
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = f64rand());
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
	for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
	for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
	for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
	for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}

void debug_compare(LeNet5 *lenet, LeNet5_fixed *lenet_fixed, image input) {
    Feature features = {0};
    Feature_fixed features_fixed = {0};

    load_input(&features, input);
    forward(lenet, &features, relu);

    load_input_fixed(&features_fixed, input);
    forward_fixed(lenet_fixed, &features_fixed);

    printf("Floating-point output:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", features.output[i]);
    }

    printf("\nFixed-point output:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", FIXED_TO_FLOAT(features_fixed.output[i]));
    }
    printf("\n");
}

void debug_layer_comparison(LeNet5 *lenet, LeNet5_fixed *lenet_fixed, image input) {
    Feature features = {0};
    Feature_fixed features_fixed = {0};

    load_input(&features, input);
    forward(lenet, &features, relu);

    load_input_fixed(&features_fixed, input);
    forward_fixed(lenet_fixed, &features_fixed);

    // Comparar cada camada
    const char* layer_names[] = {"Input", "Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Output"};
    double* float_layers[] = {
        (double*)features.input,
        (double*)features.layer1,
        (double*)features.layer2,
        (double*)features.layer3,
        (double*)features.layer4,
        (double*)features.layer5,
        features.output
    };

    fixed* fixed_layers[] = {
        (fixed*)features_fixed.input,
        (fixed*)features_fixed.layer1,
        (fixed*)features_fixed.layer2,
        (fixed*)features_fixed.layer3,
        (fixed*)features_fixed.layer4,
        (fixed*)features_fixed.layer5,
        features_fixed.output
    };

    size_t layer_sizes[] = {
        sizeof(features.input),
        sizeof(features.layer1),
        sizeof(features.layer2),
        sizeof(features.layer3),
        sizeof(features.layer4),
        sizeof(features.layer5),
        sizeof(features.output)
    };

    for (int l = 0; l < 7; l++) {
        int elements = layer_sizes[l] / sizeof(double);
        double max_diff = 0;
        double avg_diff = 0;

        for (int i = 0; i < elements; i++) {
            double fv = float_layers[l][i];
            double fxv = FIXED_TO_FLOAT(fixed_layers[l][i]);
            double diff = fabs(fv - fxv);
            avg_diff += diff;
            if (diff > max_diff) max_diff = diff;
        }
        avg_diff /= elements;

        printf("Layer %s: Max diff=%.6f, Avg diff=%.6f\n",
               layer_names[l], max_diff, avg_diff);
    }
}

void debug_first_layer_activations(LeNet5 *lenet, LeNet5_fixed *lenet_fixed, image input) {
    // Calcular a primeira camada em ponto flutuante
    Feature features = {0};
    load_input(&features, input);

    // Calcular apenas a primeira camada convolucional
    for (int oc = 0; oc < LAYER1; oc++) {
        for (int y = 0; y < LENGTH_FEATURE1; y++) {
            for (int x = 0; x < LENGTH_FEATURE1; x++) {
                double sum = 0;
                for (int ic = 0; ic < INPUT; ic++) {
                    for (int ky = 0; ky < LENGTH_KERNEL; ky++) {
                        for (int kx = 0; kx < LENGTH_KERNEL; kx++) {
                            sum += features.input[ic][y + ky][x + kx] *
                                   lenet->weight0_1[ic][oc][ky][kx];
                        }
                    }
                }
                features.layer1[oc][y][x] = relu(sum + lenet->bias0_1[oc]);
            }
        }
    }

    // Calcular a primeira camada em ponto fixo
    Feature_fixed features_fixed = {0};
    load_input_fixed(&features_fixed, input);

    for (int oc = 0; oc < LAYER1; oc++) {
        for (int y = 0; y < LENGTH_FEATURE1; y++) {
            for (int x = 0; x < LENGTH_FEATURE1; x++) {
                int32_t sum = 0;
                for (int ic = 0; ic < INPUT; ic++) {
                    for (int ky = 0; ky < LENGTH_KERNEL; ky++) {
                        for (int kx = 0; kx < LENGTH_KERNEL; kx++) {
                            sum += (int32_t)features_fixed.input[ic][y + ky][x + kx] *
                                   lenet_fixed->weight0_1[ic][oc][ky][kx];
                        }
                    }
                }
                fixed val = (fixed)(sum >> FIXED_SCALE) + lenet_fixed->bias0_1[oc];
                if (val < 0) val = 0;
                features_fixed.layer1[oc][y][x] = val;
            }
        }
    }

    // Comparar os resultados
    printf("First Layer Activations Comparison (Float vs Fixed):\n");
    printf("%10s %10s %10s %10s %10s\n", "Channel", "Y", "X", "Float", "Fixed");

    for (int oc = 0; oc < LAYER1; oc++) {
        for (int y = 0; y < LENGTH_FEATURE1; y += 4) {  // Amostrar a cada 4 pixels
            for (int x = 0; x < LENGTH_FEATURE1; x += 4) {
                double fval = features.layer1[oc][y][x];
                double fxval = FIXED_TO_FLOAT(features_fixed.layer1[oc][y][x]);

                printf("%10d %10d %10d %10.6f %10.6f", oc, y, x, fval, fxval);

                if (fabs(fval - fxval) > 0.1) {
                    printf("  *** DIFF > 0.1 ***");
                }

                printf("\n");
            }
        }
    }

    // Calcular diferenças agregadas
    double total_diff = 0;
    double max_diff = 0;
    int count = 0;

    for (int oc = 0; oc < LAYER1; oc++) {
        for (int y = 0; y < LENGTH_FEATURE1; y++) {
            for (int x = 0; x < LENGTH_FEATURE1; x++) {
                double fval = features.layer1[oc][y][x];
                double fxval = FIXED_TO_FLOAT(features_fixed.layer1[oc][y][x]);
                double diff = fabs(fval - fxval);

                total_diff += diff;
                if (diff > max_diff) max_diff = diff;
                count++;
            }
        }
    }

    printf("\nSummary:\n");
    printf("Total activations compared: %d\n", count);
    printf("Average difference: %.6f\n", total_diff / count);
    printf("Maximum difference: %.6f\n", max_diff);
    printf("Total absolute difference: %.6f\n", total_diff);
}
