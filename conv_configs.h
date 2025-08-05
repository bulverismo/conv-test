#include <stdint.h>
// Parâmetros do Acelerador (Centralizados aqui para driver e aplicação)
#define DWIDTH 8 
#define MAX_IFMAP_ROW_SIZE 32//15//32 
#define MAX_FILTER_ROW_SIZE 5//3//5
#define MAX_OFMAP_ROW_SIZE (MAX_IFMAP_ROW_SIZE - MAX_FILTER_ROW_SIZE + 1) 
typedef int16_t conv_word_t;
