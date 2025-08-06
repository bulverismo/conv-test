#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define FILE_TRAIN_IMAGE		"train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define LENET_FILE_FIXED 		"model_fixed.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000


void apply_bias_correction(LeNet5_fixed *fixed_model, LeNet5 *float_model) {
    // Apenas para a última camada (ajuste empírico)
    printf("Aplying bias correction...\n");
    for (int i = 0; i < OUTPUT; i++) {
        double float_bias = float_model->bias5_6[i];
        double current = FIXED_TO_FLOAT(fixed_model->bias5_6[i]);
        fixed_model->bias5_6[i] += FLOAT_TO_FIXED(float_bias - current);
    }
}

void convert_model(LeNet5_fixed *fixed_model, LeNet5 *float_model) {
    // 1. Converter todos os pesos
    const int total_weights = sizeof(LeNet5) / sizeof(double) - 4*sizeof(double[10]);
    for (int i = 0; i < total_weights; i++) {
        double float_val = ((double *)float_model)[i];
        ((fixed *)fixed_model)[i] = FLOAT_TO_FIXED(float_val);
    }
    
    // 2. Corrigir escala dos biases
    fixed *fixed_biases[] = {
        fixed_model->bias0_1,
        fixed_model->bias2_3,
        fixed_model->bias4_5,
        fixed_model->bias5_6
    };
    
    double *float_biases[] = {
        float_model->bias0_1,
        float_model->bias2_3,
        float_model->bias4_5,
        float_model->bias5_6
    };
    
    size_t bias_sizes[] = {LAYER1, LAYER3, LAYER5, OUTPUT};
    
    for (int layer = 0; layer < 4; layer++) {
        for (size_t i = 0; i < bias_sizes[layer]; i++) {
            // Manter os biases sem alteração de escala
            fixed_biases[layer][i] = FLOAT_TO_FIXED(float_biases[layer][i]);
        }
    }
}

void validate_conversion(LeNet5 *float_model, LeNet5_fixed *fixed_model) {
    printf("Validation:\n");

    // Verificar primeiro viés de cada camada
    printf("Conv1 bias: float=%.6f fixed=%.6f\n",
           float_model->bias0_1[0],
           FIXED_TO_FLOAT(fixed_model->bias0_1[0]));

    printf("Conv2 bias: float=%.6f fixed=%.6f\n",
           float_model->bias2_3[0],
           FIXED_TO_FLOAT(fixed_model->bias2_3[0]));

    printf("Conv3 bias: float=%.6f fixed=%.6f\n",
           float_model->bias4_5[0],
           FIXED_TO_FLOAT(fixed_model->bias4_5[0]));

    printf("FC bias: float=%.6f fixed=%.6f\n",
           float_model->bias5_6[0],
           FIXED_TO_FLOAT(fixed_model->bias5_6[0]));

    // Verificar escala relativa
    double avg_ratio = 0;
    int count = 0;

    for (int i = 0; i < OUTPUT; i++) {
        double float_val = float_model->bias5_6[i];
        double fixed_val = FIXED_TO_FLOAT(fixed_model->bias5_6[i]);

        if (fabs(float_val) > 1e-6) {
            double ratio = fixed_val / float_val;
            avg_ratio += ratio;
            count++;
            printf("%d: ratio=%.4f\n", i, ratio);
        }
    }

    avg_ratio /= count;
    printf("Average scale ratio: %.4f\n", avg_ratio);
}

int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data)*count, 1, fp_image);
	fread(label,count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}

void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
	for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
	{
		TrainBatch(lenet, train_data + i, train_label + i, batch_size);
		if (i * 100 / total_size > percent)
			printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
	}
}

int testing(LeNet5 *lenet, image *test_data, uint8 *test_label,int total_size)
{
	int right = 0, percent = 0;
	for (int i = 0; i < total_size; ++i)
	{
		uint8 l = test_label[i];
		int p = Predict(lenet, test_data[i], 10);
		right += l == p;
		if (i * 100 / total_size > percent)
			printf("test:%2d%%\n", percent = i * 100 / total_size);
	}
	return right;
}

int testing_fixed(LeNet5_fixed *lenet_fixed, image *test_data, uint8 *test_label,int total_size)
{
	int right = 0, percent = 0;
	for (int i = 0; i < total_size; ++i)
	{
		uint8 l = test_label[i];
		int p = Predict_fixed(lenet_fixed, test_data[i], 10);
		right += l == p;
		if (i * 100 / total_size > percent)
			printf("test:%2d%%\n", percent = i * 100 / total_size);
	}
	return right;
}



int save(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "wb");
	if (!fp) return 1;
	fwrite(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}


int save_fixed(LeNet5_fixed *lenet_fixed, char filename[])
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) return 1;
    fwrite(lenet_fixed, sizeof(LeNet5_fixed), 1, fp);
    fclose(fp);
    return 0;
}


int load(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}


int load_fixed(LeNet5_fixed *lenet_fixed, char filename[])
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) return 1;
    fread(lenet_fixed, sizeof(LeNet5_fixed), 1, fp);
    fclose(fp);
    return 0;
}




void foo()
{
	image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
	uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
	if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(train_data);
		free(train_label);
		system("pause");
	}
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(test_data);
		free(test_label);
		system("pause");
	}


	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
	LeNet5_fixed *lenet_fixed = (LeNet5_fixed *)malloc(sizeof(LeNet5_fixed));

	if (load(lenet, LENET_FILE)) {
		Initial(lenet);
    }

    printf("Verifying float model accuracy...\n");
    int float_accuracy = testing(lenet, test_data, test_label, 1000);
    printf("Float model accuracy on 1000 samples: %d/1000\n", float_accuracy);

    if (float_accuracy < 980) {
        printf("ERROR: Float model is not properly trained!\n");
        printf("Retraining model...\n");

        // Treinar o modelo por 1 época
        training(lenet, train_data, train_label, 300, COUNT_TRAIN);

        float_accuracy = testing(lenet, test_data, test_label, 1000);
        printf("New float accuracy: %d/1000\n", float_accuracy);

        save(lenet, LENET_FILE);

        if (float_accuracy < 980) {
            printf("FATAL ERROR: Model cannot be trained properly\n");
            exit(1);
        }
    }

  convert_model(lenet_fixed, lenet);

  apply_bias_correction(lenet_fixed, lenet);

  save_fixed(lenet_fixed, LENET_FILE_FIXED);

  printf("=== Conversion Validation ===\n");
  validate_conversion(lenet, lenet_fixed);
  debug_layer_comparison(lenet, lenet_fixed, test_data[0]);
  debug_compare(lenet, lenet_fixed, test_data[0]);

      // Debug: comparar a primeira camada
    debug_first_layer_activations(lenet, lenet_fixed, test_data[0]);

    clock_t start = clock();
    //int batches[] = { 300 };
    //for (int i = 0; i < sizeof(batches) / sizeof(*batches);++i)
    //		training(lenet, train_data, train_label, batches[i],COUNT_TRAIN);

    int right = testing_fixed(lenet_fixed, test_data, test_label, COUNT_TEST);
    printf("%d/%d\n", right, COUNT_TEST);
    printf("Time:%u\n", (unsigned)(clock() - start));
    printf("Accuracy: %d/%d (%.2f%%)\n", right, COUNT_TEST, 100.0*right/COUNT_TEST);


  
/*
    clock_t start = clock();
    int batches[] = { 300 };
    //for (int i = 0; i < sizeof(batches) / sizeof(*batches);++i)
    //		training(lenet, train_data, train_label, batches[i],COUNT_TRAIN);

    int right = testing(lenet, test_data, test_label, COUNT_TEST);
    printf("%d/%d\n", right, COUNT_TEST);
    printf("Time:%u\n", (unsigned)(clock() - start));

    */



	//save(lenet, LENET_FILE);
	free(lenet);
	free(train_data);
	free(train_label);
	free(test_data);
	free(test_label);
	system("pause");
}

int main()
{
	foo();
	return 0;
}
