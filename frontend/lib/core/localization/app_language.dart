enum AppLanguage { en, es }

extension AppLanguageX on AppLanguage {
  String get code => this == AppLanguage.en ? 'en' : 'es';
  String get label => this == AppLanguage.en ? 'English' : 'Español';

  static AppLanguage fromCode(String? code) {
    switch (code) {
      case 'es':
        return AppLanguage.es;
      case 'en':
      default:
        return AppLanguage.en;
    }
  }
}