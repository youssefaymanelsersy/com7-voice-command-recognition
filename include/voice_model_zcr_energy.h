#ifndef VOICE_MODEL_H
#define VOICE_MODEL_H

/* Generated from models/model_zcr_energy.json. Keep feature extraction in sync with features.py. */
#define VOICE_MODEL_FEATURE_COUNT 2
#define VOICE_MODEL_COMMAND_COUNT 8
#define VOICE_MODEL_K_NEIGHBORS 5
#define VOICE_MODEL_UNKNOWN_THRESHOLD 1.5f
#define VOICE_MODEL_MIN_MARGIN 0.02f

typedef struct {
    float values[VOICE_MODEL_FEATURE_COUNT];
} VoiceFeatureVector;

typedef struct {
    float mean;
    float std;
} VoiceFeatureStat;

typedef struct {
    const char* command;
    VoiceFeatureVector centroid;
    const VoiceFeatureVector* samples;
    unsigned int sample_count;
    VoiceFeatureStat command_stats[VOICE_MODEL_FEATURE_COUNT];
} VoiceCommandModel;

static const char* g_voice_model_feature_order[VOICE_MODEL_FEATURE_COUNT] = {
    "zcr",
    "energy",
};

static const float g_voice_model_feature_weights[VOICE_MODEL_FEATURE_COUNT] = {
    4.0f,
    1.8f,
};

static const float g_voice_model_feature_floors[VOICE_MODEL_FEATURE_COUNT] = {
    0.0f,
    0.0f,
};

static const VoiceFeatureStat g_voice_model_feature_stats[VOICE_MODEL_FEATURE_COUNT] = {{0.042656721f, 0.0092233053f}, {0.14804872f, 0.036156706f}};

static const VoiceFeatureVector g_voice_model_samples_on[] = {
    {{0.026792647f, 0.14795941f}},
    {{0.032480188f, 0.13888775f}},
    {{0.030127462f, 0.11695914f}},
    {{0.034124438f, 0.13950908f}},
    {{0.033993609f, 0.20880185f}},
    {{0.031902626f, 0.20334478f}},
    {{0.032216493f, 0.20565538f}},
    {{0.035668463f, 0.15502018f}},
    {{0.032459103f, 0.15524165f}},
    {{0.03597679f, 0.1462798f}},
    {{0.034442421f, 0.11995559f}},
    {{0.037891481f, 0.11552685f}},
    {{0.038666841f, 0.12586643f}},
    {{0.034789331f, 0.12510993f}},
    {{0.027604166f, 0.12622899f}},
    {{0.037467699f, 0.11363628f}},
    {{0.035948977f, 0.1239907f}},
    {{0.034607437f, 0.11582796f}},
    {{0.036970012f, 0.14578933f}},
};

static const VoiceFeatureVector g_voice_model_samples_off[] = {
    {{0.088398471f, 0.050625104f}},
    {{0.12407075f, 0.047939789f}},
    {{0.061500616f, 0.037522618f}},
    {{0.047835052f, 0.04477622f}},
    {{0.032829441f, 0.060494889f}},
    {{0.057702214f, 0.067037381f}},
    {{0.15044926f, 0.035985079f}},
    {{0.10522313f, 0.036439531f}},
    {{0.16260552f, 0.044652384f}},
    {{0.04596993f, 0.071541473f}},
    {{0.057864033f, 0.051481504f}},
    {{0.065202229f, 0.071861602f}},
    {{0.032258064f, 0.053346917f}},
    {{0.02766856f, 0.064514652f}},
    {{0.036941111f, 0.054521844f}},
    {{0.057661071f, 0.055204201f}},
    {{0.092300273f, 0.073253892f}},
    {{0.036625516f, 0.060712975f}},
    {{0.032822758f, 0.057028178f}},
    {{0.037121575f, 0.081881724f}},
};

static const VoiceFeatureVector g_voice_model_samples_start[] = {
    {{0.048557445f, 0.25579861f}},
    {{0.049568687f, 0.24366337f}},
    {{0.044860292f, 0.24451745f}},
    {{0.044294104f, 0.19933186f}},
    {{0.040220316f, 0.16872323f}},
    {{0.047132455f, 0.20553681f}},
    {{0.052071311f, 0.18141693f}},
    {{0.054303277f, 0.23059046f}},
    {{0.050308008f, 0.2138633f}},
    {{0.044762641f, 0.19561779f}},
    {{0.052115582f, 0.24706987f}},
    {{0.0439163f, 0.21891834f}},
    {{0.030912098f, 0.24165855f}},
    {{0.030751709f, 0.24725434f}},
    {{0.04266363f, 0.27219653f}},
    {{0.046052631f, 0.2910428f}},
    {{0.048262794f, 0.32850125f}},
    {{0.04410838f, 0.31656033f}},
    {{0.049028438f, 0.10156946f}},
};

static const VoiceFeatureVector g_voice_model_samples_stop[] = {
    {{0.035815511f, 0.14949451f}},
    {{0.039898023f, 0.15594095f}},
    {{0.038046274f, 0.15435275f}},
    {{0.034526691f, 0.14241998f}},
    {{0.034251869f, 0.15923536f}},
    {{0.034127086f, 0.3017219f}},
    {{0.033120204f, 0.27545261f}},
    {{0.035040773f, 0.25098696f}},
    {{0.032820251f, 0.24062406f}},
    {{0.03256705f, 0.21295297f}},
    {{0.035239447f, 0.22708575f}},
    {{0.033260841f, 0.26109007f}},
    {{0.036123123f, 0.32827723f}},
    {{0.034403671f, 0.32527518f}},
    {{0.034201123f, 0.29415235f}},
    {{0.032579534f, 0.30940345f}},
    {{0.032479938f, 0.32366064f}},
    {{0.036206018f, 0.32330897f}},
    {{0.036333505f, 0.24188037f}},
};

static const VoiceFeatureVector g_voice_model_samples_left[] = {
    {{0.027785199f, 0.042151414f}},
    {{0.057236534f, 0.043926075f}},
    {{0.055231292f, 0.036442652f}},
    {{0.041657958f, 0.055979762f}},
    {{0.026155021f, 0.034077395f}},
    {{0.042198721f, 0.040288713f}},
    {{0.033159558f, 0.036779482f}},
    {{0.050690468f, 0.052087918f}},
    {{0.041424602f, 0.053174041f}},
    {{0.039132793f, 0.041758906f}},
    {{0.045695979f, 0.04575377f}},
    {{0.040497337f, 0.052826319f}},
    {{0.033724781f, 0.046125107f}},
    {{0.051525425f, 0.04162427f}},
    {{0.037858792f, 0.04778007f}},
    {{0.032227978f, 0.049365543f}},
    {{0.038445108f, 0.049783394f}},
    {{0.051991615f, 0.032670863f}},
    {{0.028102491f, 0.041117158f}},
    {{0.028826838f, 0.05229979f}},
};

static const VoiceFeatureVector g_voice_model_samples_right[] = {
    {{0.042764857f, 0.13487253f}},
    {{0.039741937f, 0.13668278f}},
    {{0.043483939f, 0.13561708f}},
    {{0.040795382f, 0.13617441f}},
    {{0.036728948f, 0.16649272f}},
    {{0.0428553f, 0.14761525f}},
    {{0.032778423f, 0.1426944f}},
    {{0.041391801f, 0.15684757f}},
    {{0.0336156f, 0.17245759f}},
    {{0.055490728f, 0.1712939f}},
    {{0.038451746f, 0.25029737f}},
    {{0.034738187f, 0.24556418f}},
    {{0.042697784f, 0.25315335f}},
    {{0.039974459f, 0.26386166f}},
    {{0.037871033f, 0.19655982f}},
    {{0.041446093f, 0.20321096f}},
    {{0.040463917f, 0.22621332f}},
    {{0.035265453f, 0.2262276f}},
};

static const VoiceFeatureVector g_voice_model_samples_up[] = {
    {{0.024280068f, 0.10192767f}},
    {{0.023237018f, 0.070719257f}},
    {{0.02545034f, 0.090460539f}},
    {{0.024419062f, 0.059868447f}},
    {{0.025995316f, 0.052876607f}},
    {{0.025756337f, 0.083726272f}},
    {{0.024169184f, 0.086661234f}},
    {{0.025917927f, 0.06641084f}},
    {{0.022593681f, 0.093485773f}},
    {{0.02867806f, 0.052673794f}},
    {{0.026334776f, 0.075259387f}},
    {{0.024760077f, 0.069486476f}},
    {{0.022213927f, 0.072575085f}},
    {{0.023307437f, 0.074999973f}},
    {{0.022087244f, 0.050107673f}},
    {{0.026528258f, 0.060507722f}},
    {{0.022133939f, 0.059943508f}},
    {{0.020501139f, 0.052811291f}},
    {{0.029989213f, 0.041393653f}},
    {{0.020066889f, 0.068591028f}},
};

static const VoiceFeatureVector g_voice_model_samples_down[] = {
    {{0.039942939f, 0.15431435f}},
    {{0.044756867f, 0.27656853f}},
    {{0.052897707f, 0.32613471f}},
    {{0.052229133f, 0.33711913f}},
    {{0.054168813f, 0.31849596f}},
    {{0.056734852f, 0.30790472f}},
    {{0.051480051f, 0.22979081f}},
    {{0.041564792f, 0.23250581f}},
    {{0.058088616f, 0.24962917f}},
    {{0.052544106f, 0.21873195f}},
    {{0.051288694f, 0.17186776f}},
    {{0.057363126f, 0.1698879f}},
    {{0.067146897f, 0.19844544f}},
    {{0.061134234f, 0.16080412f}},
    {{0.06132989f, 0.17941935f}},
    {{0.052222367f, 0.13871676f}},
    {{0.057902168f, 0.17708839f}},
    {{0.061729997f, 0.20225316f}},
    {{0.054589931f, 0.15344447f}},
    {{0.048021864f, 0.1665338f}},
};

static const VoiceCommandModel g_voice_model_commands[VOICE_MODEL_COMMAND_COUNT] = {
    {
        "on",
        {{0.033901589f, 0.14366269f}},
        g_voice_model_samples_on,
        (unsigned int)(sizeof(g_voice_model_samples_on) / sizeof(g_voice_model_samples_on[0])),
        {{0.033901589f, 0.0031582092f}, {0.14366269f, 0.029979965f}},
    },
    {
        "off",
        {{0.067652479f, 0.056041098f}},
        g_voice_model_samples_off,
        (unsigned int)(sizeof(g_voice_model_samples_off) / sizeof(g_voice_model_samples_off[0])),
        {{0.067652479f, 0.039256868f}, {0.056041098f, 0.012653955f}},
    },
    {
        "start",
        {{0.0454679f, 0.23178059f}},
        g_voice_model_samples_start,
        (unsigned int)(sizeof(g_voice_model_samples_start) / sizeof(g_voice_model_samples_start[0])),
        {{0.0454679f, 0.0061122035f}, {0.23178059f, 0.05137372f}},
    },
    {
        "stop",
        {{0.034791628f, 0.24617453f}},
        g_voice_model_samples_stop,
        (unsigned int)(sizeof(g_voice_model_samples_stop) / sizeof(g_voice_model_samples_stop[0])),
        {{0.034791628f, 0.0019155066f}, {0.24617453f, 0.065374741f}},
    },
    {
        "left",
        {{0.040178424f, 0.044800632f}},
        g_voice_model_samples_left,
        (unsigned int)(sizeof(g_voice_model_samples_left) / sizeof(g_voice_model_samples_left[0])),
        {{0.040178424f, 0.0092834624f}, {0.044800632f, 0.0066470903f}},
    },
    {
        "right",
        {{0.040030866f, 0.18699092f}},
        g_voice_model_samples_right,
        (unsigned int)(sizeof(g_voice_model_samples_right) / sizeof(g_voice_model_samples_right[0])),
        {{0.040030866f, 0.00494932f}, {0.18699092f, 0.045293592f}},
    },
    {
        "up",
        {{0.024420994f, 0.069224311f}},
        g_voice_model_samples_up,
        (unsigned int)(sizeof(g_voice_model_samples_up) / sizeof(g_voice_model_samples_up[0])),
        {{0.024420994f, 0.0024590842f}, {0.069224311f, 0.015677857f}},
    },
    {
        "down",
        {{0.053856852f, 0.21848282f}},
        g_voice_model_samples_down,
        (unsigned int)(sizeof(g_voice_model_samples_down) / sizeof(g_voice_model_samples_down[0])),
        {{0.053856852f, 0.0066517883f}, {0.21848282f, 0.062252726f}},
    },
};

#endif
