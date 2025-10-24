import React from 'react';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { 
  FaHome, 
  FaKey, 
  FaChartBar, 
  FaFileContract,
  FaHandshake,
  FaShieldAlt,
  FaGlobe,
  FaClock
} from 'react-icons/fa';
import './Services.css';

const Services = () => {
  const [ref, inView] = useInView({
    threshold: 0.1,
    triggerOnce: true
  });

  const services = [
    {
      icon: FaHome,
      title: 'Продажа недвижимости',
      description: 'Поможем продать вашу недвижимость по максимальной цене с полным юридическим сопровождением',
      features: ['Оценка недвижимости', 'Маркетинговая стратегия', 'Показы объектов', 'Ведение переговоров']
    },
    {
      icon: FaKey,
      title: 'Аренда',
      description: 'Найдем идеальную аренду для ваших потребностей среди лучших объектов Дубая',
      features: ['Поиск по критериям', 'Организация просмотров', 'Помощь в оформлении', 'Контроль договоров']
    },
    {
      icon: FaChartBar,
      title: 'Инвестиции',
      description: 'Консультации по инвестициям в недвижимость с анализом рынка и прогнозами',
      features: ['Анализ рынка', 'Инвестиционные стратегии', 'ROI расчеты', 'Риск-менеджмент']
    },
    {
      icon: FaFileContract,
      title: 'Юридическое сопровождение',
      description: 'Полное юридическое сопровождение сделок с недвижимостью',
      features: ['Проверка документов', 'Составление договоров', 'Регистрация сделок', 'Консультации']
    },
    {
      icon: FaHandshake,
      title: 'Управление недвижимостью',
      description: 'Профессиональное управление вашей недвижимостью',
      features: ['Поиск арендаторов', 'Сбор арендной платы', 'Техническое обслуживание', 'Отчетность']
    },
    {
      icon: FaShieldAlt,
      title: 'Страхование',
      description: 'Страхование недвижимости и помощь в оформлении страховых случаев',
      features: ['Подбор страховки', 'Оформление полисов', 'Урегулирование убытков', 'Консультации']
    }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 50, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.6,
        ease: "easeOut"
      }
    }
  };

  return (
    <section id="services" className="services">
      <div className="container">
        <motion.div
          className="section-header"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="section-title">Наши услуги</h2>
          <p className="section-subtitle">
            Полный спектр услуг в сфере недвижимости Дубая
          </p>
        </motion.div>

        <motion.div
          className="services-grid"
          ref={ref}
          variants={containerVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
        >
          {services.map((service, index) => (
            <motion.div
              key={index}
              className="service-card"
              variants={itemVariants}
              whileHover={{ y: -10 }}
            >
              <div className="service-icon-container">
                <motion.div
                  className="service-icon"
                  whileHover={{ 
                    scale: 1.1,
                    rotate: 5,
                    transition: { duration: 0.3 }
                  }}
                >
                  <service.icon />
                </motion.div>
              </div>
              
              <h3 className="service-title">{service.title}</h3>
              <p className="service-description">{service.description}</p>
              
              <ul className="service-features">
                {service.features.map((feature, featureIndex) => (
                  <li key={featureIndex} className="feature-item">
                    <span className="feature-bullet" />
                    {feature}
                  </li>
                ))}
              </ul>

              <motion.button
                className="service-btn"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Узнать подробнее
              </motion.button>
            </motion.div>
          ))}
        </motion.div>

        <motion.div
          className="services-cta"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          viewport={{ once: true }}
        >
          <div className="cta-content">
            <h3>Нужна консультация?</h3>
            <p>Наши эксперты готовы помочь вам с любыми вопросами по недвижимости</p>
            <motion.button
              className="btn btn-primary"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => {
                const contactSection = document.querySelector('#contact');
                if (contactSection) {
                  contactSection.scrollIntoView({ behavior: 'smooth' });
                }
              }}
            >
              Получить консультацию
            </motion.button>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Services;
