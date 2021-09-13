package com.pi.stela.main;

/**
 * Created by Paul on 08/10/2016.
 */
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;

import com.pi.stela.R;
import com.pi.stela.network.RedisConnector;
import com.pi.stela.sensors.*;

import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.support.v4.app.Fragment;
import android.util.Log;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.View.OnClickListener;
import android.view.ViewGroup.LayoutParams;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ImageView.ScaleType;

public class StelaActivity extends Fragment {

    public static TextToSpeech mTts;
    public static boolean TTSInitialized = false;
    private View mLeak; // retained
    private boolean AlreadyBuild = false;
    private boolean AlreadyListening = false;
    private final String host = "149.202.55.170";
    private final int port = 6379;
    private final RedisConnector redisConnector = new RedisConnector(host, port);
    @Override
    public void onDestroyView() {
        super.onDestroyView();
        mLeak = getView(); // now cleaning up!
        System.out.println("JE SUIS DESTROY");
    }
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setRetainInstance(true);
    }
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {

        if(!AlreadyBuild){
            AlreadyBuild=true;
            final MainActivity MN = MainActivity.MN;
            RelativeLayout MainStelaRL = new RelativeLayout(MN);

            //Instanciations des proportions
            final double S = MainActivity.S;
            final double W = MainActivity.W;
            final double H = MainActivity.H;

            //Initialisation du nombre d'or
            final double GoldNumber = (double)((1 + Math.sqrt(5))/2);

            //Initialisation des polices
            Typeface RobotoFont = Typeface.createFromAsset(MN.getAssets(), "robotoregular.ttf");

		/*
		 *
		 * 								TextViews
		 *
		 */

            //Initialisation des TextViews
            TextView WelcomeDisplay = new TextView(MN);

            //Récupération de la date et de l'heure actuelle
            Calendar MainCalendar = Calendar.getInstance();
            SimpleDateFormat TimeFormat = new SimpleDateFormat("HH:mm");
            SimpleDateFormat DateFormat = new SimpleDateFormat("EEE dd MMM yyyy");
            String CurrentTime = TimeFormat.format(MainCalendar.getTime());
            String CurrentDate = DateFormat.format(MainCalendar.getTime());

            //Récupération de la température et du taux d'humidité
            String CurrentTemperature = TemperatureSensor.CurrentTemperature;
            String CurrentHumidity = HumiditySensor.CurrentHumidity;

            //Application de leur texte aux TextViews
            WelcomeDisplay.setText("Bonjour, comment-allez vous ?");

            //Application de leur police aux TextViews
            WelcomeDisplay.setTypeface(RobotoFont);

            //Application de la taille de leur police aux TextViews
            WelcomeDisplay.setTextSize((float) (GoldNumber * 35 * S));

            //Application de la couleur de la police des TextViews
            WelcomeDisplay.setTextColor(Color.parseColor("#FFFFFF"));

            //Initialisation des paramètres des TextViews
            RelativeLayout.LayoutParams WelcomeDisplayParams = new
                    RelativeLayout.LayoutParams(LayoutParams.MATCH_PARENT, (int) (70 * GoldNumber * H));

            //Application des marges aux paramètres des TextViews
            WelcomeDisplayParams.setMargins(0, (int) (700 * GoldNumber * H), 0, 0);

            //Application de leur paramètre aux TextViews
            WelcomeDisplay.setLayoutParams(WelcomeDisplayParams);

            //Application de la gravité dans les TextViews
            WelcomeDisplay.setGravity(Gravity.CENTER_HORIZONTAL);

            //Mise en place des TextViews dans le Layout principal
            MainStelaRL.addView(WelcomeDisplay);

		/*
		 *
		 *								Buttons
		 *
		 */

            //Initialisation des Buttons
            final Button SettingsDisplay = new Button(MN);

            //Application des Backgrounds aux Buttons
            SettingsDisplay.setBackgroundColor(Color.TRANSPARENT);

            //Distribution de leur texte aux Buttons
            SettingsDisplay.setText("Historique et paramètres");

            //Application des polices aux Buttons
            SettingsDisplay.setTypeface(RobotoFont);

            //Application des tailles des polices aux Buttons
            SettingsDisplay.setTextSize((float) (GoldNumber * 20 * S));

            //Application des couleurs de polices aux Buttons
            SettingsDisplay.setTextColor(Color.parseColor("#336699"));

            //Application de la gravité dans les Buttons
            SettingsDisplay.setGravity(Gravity.CENTER);

            //Application des marges (paddings) aux buttons
            SettingsDisplay.setPadding(0, 0, 0, 0);

            //Initialisation des paramètres des Buttons
            RelativeLayout.LayoutParams SettingsDisplayParams = new
                    RelativeLayout.LayoutParams((int) (300 * GoldNumber * W), (int) (50 * GoldNumber * H));

            //Application des marges aux paramètres des Buttons
            SettingsDisplayParams.setMargins((int) (((W * 1080 - (300 * GoldNumber * W))/2)), (int) (800 * GoldNumber * H), 0, 0);

            //Application de leur paramètre aux Buttons
            SettingsDisplay.setLayoutParams(SettingsDisplayParams);

            //Application des OnClickListener aux Buttons
            SettingsDisplay.setOnClickListener(new OnClickListener(){
                @Override
                public void onClick(View v) {
                }
            });

            //Mise en place des Buttons dans le Layout principal
            MainStelaRL.addView(SettingsDisplay);

		/*
		 *
		 *								ImageButtons
		 *
		 */

            //Initialisation des ImageButtons
            final ImageButton StelaViewDisplay = new ImageButton(MN);

            //Application des Backgrounds aux ImageButtons
            StelaViewDisplay.setBackgroundColor(Color.TRANSPARENT);

            //Distribution de leur image aux ImageButtons
            StelaViewDisplay.setImageResource(R.drawable.stela_view);

            //Application des paramètres de contenance des images aux ImageButtons
            StelaViewDisplay.setScaleType(ScaleType.FIT_CENTER);

            //Initialisation des paramètres des ImageButtons
            final RelativeLayout.LayoutParams StelaViewDisplayParams = new
                    RelativeLayout.LayoutParams((int) (200 * GoldNumber * W), (int) (200 * GoldNumber * H));

            //Application des marges aux paramètres des ImageButtons
            StelaViewDisplayParams.setMargins(0, (int) (450 * GoldNumber * H), 0, 0);

            //Application des règles aux paramètres
            StelaViewDisplayParams.addRule(RelativeLayout.CENTER_HORIZONTAL);

            //Application de leur paramètre aux ImageButtons
            StelaViewDisplay.setLayoutParams(StelaViewDisplayParams);

            //Instanciation du Speech Recognizer
            final SpeechRecognizer sr = SpeechRecognizer.createSpeechRecognizer(MN);

            //Application de leur OnClickListener aux ImageButtons
            StelaViewDisplay.setOnClickListener(new OnClickListener(){
                @Override
                public void onClick(View v) {

                    if(!AlreadyListening){
                        sr.setRecognitionListener(new RecognitionListener(){
                            @Override
                            public void onReadyForSpeech(Bundle params) {
                                Toast t = Toast.makeText(MN, "Ready to listen!", Toast.LENGTH_SHORT);
                                t.show();
                            }
                            @Override
                            public void onBeginningOfSpeech() {
                                Toast t = Toast.makeText(MN, "You just began", Toast.LENGTH_SHORT);
                                t.show();
                            }
                            @Override
                            public void onRmsChanged(float rmsdB) {
                                double size = 0.04 * rmsdB + 1.12;
                                System.out.println("RMSCHANGED " + size);
                                RelativeLayout.LayoutParams StelaViewDisplayParamsSecond = new
                                        RelativeLayout.LayoutParams((int) (200 * GoldNumber * W * size), (int) (200 * GoldNumber * H * size));
                                StelaViewDisplayParamsSecond.setMargins(0, (int) (450 * GoldNumber * H - ((200 * GoldNumber * H * size)-(200 * GoldNumber * H))/2), 0, 0);
                                StelaViewDisplayParamsSecond.addRule(RelativeLayout.CENTER_HORIZONTAL);
                                StelaViewDisplay.setLayoutParams(StelaViewDisplayParamsSecond);
                                if(!AlreadyListening){
                                    StelaViewDisplay.setLayoutParams(StelaViewDisplayParams);
                                }
                            }
                            @Override
                            public void onBufferReceived(byte[] buffer) {
                            }
                            @Override
                            public void onEndOfSpeech() {
                                AlreadyListening = !AlreadyListening;
                                Toast t = Toast.makeText(MN, "That's the End", Toast.LENGTH_SHORT);
                                t.show();
                            }
                            @Override
                            public void onError(int error) {

                                Log.d("STELA", "onError " + error);
                            }
                            @Override
                            public void onResults(Bundle results) {
                                String str = new String();
                                Log.d("STELA", "onResults " + results);
                                ArrayList<String> data = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                                for (int i = 0; i < data.size(); i++){
                                    Log.d("STELA", "result " + data.get(i));
                                    str += data.get(i);
                                }
                                if(data.size() > 0){
                                    String result = data.get(0);
                                    System.out.println((CharSequence) data.get(0));
                                    Toast t = Toast.makeText(MN, "Result: " + result, Toast.LENGTH_SHORT);
                                    t.show();
                                    if(TTSInitialized){
                                        redisConnector.queryAI(result, mTts);
                                    }else{
                                        Intent checkIntent = new Intent();
                                        checkIntent.setAction(TextToSpeech.Engine.ACTION_CHECK_TTS_DATA);
                                        startActivityForResult(checkIntent, 0x01);
                                    }
                                }
                            }
                            @Override
                            public void onPartialResults(Bundle partialResults) {
                                Toast.makeText(MN, "Trop Fort !", Toast.LENGTH_SHORT).show();
                                if(partialResults.containsKey("Stella")){
                                    Toast t = Toast.makeText(MN, "Trop Fort !", Toast.LENGTH_SHORT);
                                    t.show();
                                }
                            }
                            @Override
                            public void onEvent(int eventType, Bundle params) {
                                Toast t = Toast.makeText(MN, "Event", Toast.LENGTH_SHORT);
                                t.show();
                            }
                        });
                        Intent recognizerIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
                        recognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
                        recognizerIntent.putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, "voice.recognition.test");
                        recognizerIntent.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 5);
                        recognizerIntent.putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true);
                        recognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, "fr-FR");
                        recognizerIntent.putExtra(RecognizerIntent.EXTRA_SUPPORTED_LANGUAGES, "en-EN");
                        sr.startListening(recognizerIntent);
                        AlreadyListening = !AlreadyListening;
                    }else{
                        sr.stopListening();
                        AlreadyListening = !AlreadyListening;
                        StelaViewDisplay.setLayoutParams(StelaViewDisplayParams);
                    }
                }
            });

            //Mise en place des ImageButtons dans le Layout principal
            MainStelaRL.addView(StelaViewDisplay);

		/*
		 *
		 *								EditTexts
		 *
		 */

            //Initialisation des EditTexts
            EditText AdviceDisplay = new EditText(MN);

            //Application des Backgrounds aux EditTexts
            AdviceDisplay.setBackgroundResource(R.drawable.main_edit_text);

            //Modification des Paddings
            AdviceDisplay.setPadding(0, 0, 0, 0);

            //Application de leur hint aux EditTexts
            AdviceDisplay.setHint("Essayez \"Stela, appelle Claire\"");

            //Application de leur police aux EditTexts
            AdviceDisplay.setTypeface(RobotoFont);

            //Application de la taille de leur police aux EditTexts
            AdviceDisplay.setTextSize((float) (GoldNumber * 35 * S));

            //Application de la couleur de la police des EditTexts
            AdviceDisplay.setTextColor(Color.parseColor("#FFFFFF"));
            AdviceDisplay.setHintTextColor(Color.parseColor("#FFFFFF"));

            //Initialisation des paramètres des EditTexts
            RelativeLayout.LayoutParams AdviceDisplayParams = new
                    RelativeLayout.LayoutParams((int) (600 * GoldNumber * W), (int) (80 * GoldNumber * H));

            //Application des marges aux paramètres des EditTexts
            AdviceDisplayParams.setMargins(0, (int) (1050 * GoldNumber * H), 0, 0);

            //Application des règles aux paramètres
            AdviceDisplayParams.addRule(RelativeLayout.CENTER_HORIZONTAL);

            //Application de leur paramètre aux EditTexts
            AdviceDisplay.setLayoutParams(AdviceDisplayParams);

            //Application de la gravité dans les EditTexts
            AdviceDisplay.setGravity(Gravity.CENTER);

            //Mise en place des EditTexts dans le Layout principal
            MainStelaRL.addView(AdviceDisplay);
            return MainStelaRL;
        }else{
            return mLeak;
        }
    }
}

